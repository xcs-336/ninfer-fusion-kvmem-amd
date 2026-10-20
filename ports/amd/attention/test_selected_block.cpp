#include "ninfer/ops/selected_block_attention.h"
#include "ops/op_tester.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {
constexpr int Dim = 256;
constexpr int Heads = 24;
constexpr int KVHeads = 2;
constexpr std::size_t Guard = 64;
// Same reduction and BF16 gross-error policy as causal BF16 attention. The
// oracle starts after the represented cache boundary for both storage types;
// FP8 values expand exactly to BF16 in this particular production Op.
constexpr ReductionCriterion Criterion{2.8e-3, 1e-3, kBf16GrossRelativeFloor};

unsigned next(unsigned& state) {
    state = state * 1664525U + 1013904223U;
    return state;
}

double fp8_value(std::uint8_t bits) {
    const int exponent = (bits >> 3) & 15;
    const int fraction = bits & 7;
    const double magnitude = exponent == 0 ? std::ldexp(static_cast<double>(fraction), -9)
                                          : std::ldexp(1.0 + fraction / 8.0, exponent - 7);
    return (bits & 128) != 0 ? -magnitude : magnitude;
}

struct HostPlane {
    DType type;
    std::vector<std::uint8_t> bits;

    HostPlane(DType format, std::size_t elements, unsigned seed)
        : type(format), bits(elements * (format == DType::BF16 ? 2 : 1)) {
        for (std::size_t i = 0; i < elements; ++i) {
            const unsigned random = next(seed);
            if (type == DType::BF16) {
                const float value = static_cast<float>(static_cast<int>(random % 65) - 32) / 32.0F;
                const auto code = f32_to_bf16(value);
                bits[2 * i] = static_cast<std::uint8_t>(code);
                bits[2 * i + 1] = static_cast<std::uint8_t>(code >> 8);
            } else {
                // Finite represented E4M3 values, including signs and varying
                // binades. No production codec is used by this host oracle.
                bits[i] = static_cast<std::uint8_t>((random & 128U) | (8U + (random >> 8) % 56U));
            }
        }
    }

    double value(std::size_t index) const {
        if (type == DType::BF16) {
            const std::uint16_t code = bits[2 * index] | (std::uint16_t(bits[2 * index + 1]) << 8);
            return bf16_to_f32(code);
        }
        return fp8_value(bits[index]);
    }
};

struct Stream {
    hipStream_t value{};
    Stream() { cuda_check(hipStreamCreate(&value), "hipStreamCreate"); }
    ~Stream() { if (value != nullptr) { (void)hipStreamDestroy(value); } }
};

template <class T>
int readonly(const char* name, const DeviceBuffer& device, const std::vector<T>& original) {
    if (from_device<T>(device, original.size()) == original) { return 0; }
    std::fprintf(stderr, "selected attention changed %s\n", name);
    return 1;
}

int run(DType type, bool shared_row, int rows, bool dense, bool empty, int forced_position,
        bool graph) {
    std::vector<int> positions(rows), table_rows(rows), counts(rows);
    std::vector<int> selections(static_cast<std::size_t>(512) * rows, -1);
    constexpr int Positions[] = {0, 3, 4, 63, 64, 67, 127, 129, 191, 193, 255};
    for (int row = 0; row < rows; ++row) {
        positions[row] = empty ? 3 : forced_position >= 0 ? forced_position : Positions[row % 11];
        table_rows[row] = shared_row ? 1 : row % 2;
        const int complete = (positions[row] + 1) / 4;
        std::vector<int> blocks;
        for (int block = 0; block < complete && !empty; ++block) {
            if (dense || (block + row) % 3 == 0 || block == complete - 1) { blocks.push_back(block); }
        }
        // The contract allows any order; this also exercises the dense route's
        // mathematically equivalent canonical ordering.
        std::reverse(blocks.begin(), blocks.end());
        counts[row] = static_cast<int>(blocks.size());
        std::copy(blocks.begin(), blocks.end(), selections.begin() + row * 512);
    }
    const int logical_pages = *std::max_element(positions.begin(), positions.end()) / 64 + 1;
    const int physical_pages = 2 * logical_pages;
    std::vector<int> tables(2 * logical_pages);
    for (int row = 0; row < 2; ++row) {
        for (int page = 0; page < logical_pages; ++page) {
            tables[row * logical_pages + page] = (2 * page + row + physical_pages - 1) % physical_pages;
        }
    }
    const std::size_t elements = static_cast<std::size_t>(physical_pages) * KVHeads * 64 * Dim;
    HostPlane keys(type, elements, 76541U), values(type, elements, 99117U);
    unsigned seed = 37189U;
    std::vector<std::uint16_t> query(static_cast<std::size_t>(rows) * Heads * Dim);
    for (auto& code : query) {
        code = f32_to_bf16(static_cast<float>(static_cast<int>(next(seed) % 65) - 32) / 32.0F);
    }

    std::vector<double> expected(query.size(), 0.0);
    for (int row = 0; row < rows; ++row) {
        std::vector<int> visible;
        for (int block = 0; block < counts[row]; ++block) {
            for (int token = 0; token < 4; ++token) { visible.push_back(4 * selections[row * 512 + block] + token); }
        }
        for (int token = ((positions[row] + 1) / 4) * 4; token <= positions[row]; ++token) {
            visible.push_back(token);
        }
        if (visible.empty()) { continue; }
        for (int head = 0; head < Heads; ++head) {
            std::vector<double> scores(visible.size());
            double maximum = -std::numeric_limits<double>::infinity();
            const std::size_t qbase = (static_cast<std::size_t>(row) * Heads + head) * Dim;
            const auto offset = [&](int token) {
                const int physical = tables[table_rows[row] * logical_pages + token / 64];
                return ((static_cast<std::size_t>(physical) * KVHeads + head / 12) * 64 + token % 64) * Dim;
            };
            for (std::size_t key = 0; key < visible.size(); ++key) {
                const auto base = offset(visible[key]);
                double dot = 0.0;
                for (int dim = 0; dim < Dim; ++dim) { dot += double(bf16_to_f32(query[qbase + dim])) * keys.value(base + dim); }
                scores[key] = dot / 16.0;
                maximum = std::max(maximum, scores[key]);
            }
            double denominator = 0.0;
            for (auto& score : scores) { score = std::exp(score - maximum); denominator += score; }
            for (int dim = 0; dim < Dim; ++dim) {
                double sum = 0.0;
                for (std::size_t key = 0; key < visible.size(); ++key) { sum += scores[key] * values.value(offset(visible[key]) + dim); }
                expected[qbase + dim] = sum / denominator;
            }
        }
    }

    auto d_query = to_device(query);
    auto d_positions = to_device(positions);
    auto d_rows = to_device(table_rows);
    auto d_counts = to_device(counts);
    auto d_selections = to_device(selections);
    auto d_tables = to_device(tables);
    auto d_keys = to_device(keys.bits);
    auto d_values = to_device(values.bits);
    const std::size_t output_bytes = query.size() * sizeof(std::uint16_t);
    std::vector<std::uint8_t> guarded(Guard + output_bytes + Guard, 0xA5);
    auto d_output = to_device(guarded);
    auto* output = static_cast<std::uint8_t*>(d_output.p) + Guard;
    Tensor q(d_query.p, DType::BF16, {Dim, Heads, rows});
    Tensor p(d_positions.p, DType::I32, {rows});
    Tensor row_ids(d_rows.p, DType::I32, {rows});
    Tensor count(d_counts.p, DType::I32, {rows});
    Tensor selected(d_selections.p, DType::I32, {512, rows});
    Tensor out(output, DType::BF16, {Dim, Heads, rows});
    ops::SelectedBlockAttentionCache cache{
        Tensor(d_keys.p, type, {Dim, 64, KVHeads, physical_pages}),
        Tensor(d_values.p, type, {Dim, 64, KVHeads, physical_pages}),
        Tensor(d_tables.p, DType::I32, {logical_pages, 2})};
    // The shared-row overload has no workspace. DeviceArena deliberately
    // rejects an owning zero-byte allocation, so construct it only for the
    // batch overload that actually consumes transient storage.
    std::optional<WorkspaceArena> workspace;
    if (!shared_row) { workspace.emplace(ops::selected_block_attention_workspace_capacity_bytes(rows)); }
    Stream stream;
    const auto launch = [&] {
        if (shared_row) {
            ops::selected_block_attention(q, p, 1, selected, count, cache, out, stream.value);
        } else {
            ops::selected_block_attention(q, p, row_ids, selected, count, cache, *workspace, out, stream.value);
        }
    };
    if (graph) {
        hipGraph_t definition{};
        hipGraphExec_t executable{};
        cuda_check(hipStreamBeginCapture(stream.value, hipStreamCaptureModeThreadLocal), "begin capture");
        launch();
        cuda_check(hipStreamEndCapture(stream.value, &definition), "end capture");
        cuda_check(hipGraphInstantiateWithFlags(&executable, definition, 0), "instantiate graph");
        cuda_check(hipGraphLaunch(executable, stream.value), "launch graph");
        cuda_check(hipStreamSynchronize(stream.value), "graph synchronize");
        cuda_check(hipGraphExecDestroy(executable), "destroy executable");
        cuda_check(hipGraphDestroy(definition), "destroy graph");
    } else {
        launch();
        cuda_check(hipStreamSynchronize(stream.value), "synchronize");
    }
    const std::string label = std::string("selected_attention/") + (type == DType::BF16 ? "bf16/" : "fp8/") +
        (shared_row ? "shared/" : "batch/") + std::to_string(rows) +
        (empty ? "/empty" : dense ? "/dense" : "/sparse") + (graph ? "/graph" : "");
    int failures = verify_reduction(label, from_device_bf16(output, query.size()), expected, Criterion);
    const auto actual_guarded = from_device<std::uint8_t>(d_output, guarded.size());
    if (!std::equal(actual_guarded.begin(), actual_guarded.begin() + Guard, guarded.begin()) ||
        !std::equal(actual_guarded.end() - Guard, actual_guarded.end(), guarded.end() - Guard)) {
        std::fprintf(stderr, "%s output guard changed\n", label.c_str());
        ++failures;
    }
    if (empty) {
        const auto zeros = from_device<std::uint16_t>(output, query.size());
        if (std::any_of(zeros.begin(), zeros.end(), [](auto value) { return value != 0; })) { ++failures; }
    }
    failures += readonly("query", d_query, query) + readonly("positions", d_positions, positions) +
        readonly("table rows", d_rows, table_rows) + readonly("counts", d_counts, counts) +
        readonly("selections", d_selections, selections) + readonly("tables", d_tables, tables) +
        readonly("keys", d_keys, keys.bits) + readonly("values", d_values, values.bits);
    return failures;
}
} // namespace

int main() {
    try {
        int failures = 0;
        for (const auto type : {DType::BF16, DType::FP8_E4M3FN}) {
            failures += run(type, false, 1, false, true, -1, false);
            failures += run(type, false, 8, false, false, -1, true);
            failures += run(type, false, 8, true, false, -1, false);
            failures += run(type, true, 1, false, true, -1, false);
            failures += run(type, true, 3, false, false, -1, true);
            failures += run(type, true, 17, false, false, -1, false);
            failures += run(type, true, 17, true, false, -1, false);
            failures += run(type, true, 1, true, false, 2047, false);
        }
        std::printf("selected-block public attention FP64 oracle: %s\n", failures == 0 ? "PASS" : "FAIL");
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "selected attention: %s\n", error.what());
        return 1;
    }
}
