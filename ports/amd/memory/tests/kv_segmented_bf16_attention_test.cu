// SOURCE ONLY candidate, not built/qualified. Actual original native append,
// small-T BF16 WMMA+reducer and prompt BF16 WMMA bodies are instantiated through
// segmented_bf16_launch.cuh. The only toy GPU kernel is the raw-span BORROWER.
// This does not enable Attention/Engine or claim production allocator integration.
#include "amd-memory/segmented_bf16_launch.cuh"
#include "amd-memory/kv_segmented_pool_owner.h"
#include "core/host_kv_arena.h"
#include "op_check.h"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using namespace ninfer;
using namespace ninfer::amd::memory;
constexpr std::size_t GuardBytes = 256;
constexpr std::uint8_t Guard = 0xa5;
constexpr int Groups = 8, LogicalPages = 3, LogicalTokens = LogicalPages * 64, D = 256;
constexpr float Scale = 0.0625f;

void need(bool value, const char* text) { if (!value) throw std::runtime_error(text); }
void check(hipError_t value, const char* where) {
    if (value != hipSuccess) throw std::runtime_error(std::string(where) + ": " + hipGetErrorString(value));
}

class Stream {
public:
    Stream() { check(hipStreamCreateWithFlags(&value, hipStreamNonBlocking), "create test stream"); }
    ~Stream() { if (value) { (void)hipStreamSynchronize(value); (void)hipStreamDestroy(value); } }
    Stream(const Stream&) = delete;
    hipStream_t value = nullptr;
};

class Buffer {
public:
    explicit Buffer(std::size_t bytes) : bytes(bytes) {
        need(bytes != 0 && bytes <= std::numeric_limits<std::size_t>::max() - 2 * GuardBytes,
             "test buffer length overflow");
        check(hipMalloc(&allocation, bytes + 2 * GuardBytes), "test guarded allocation");
        data = static_cast<std::uint8_t*>(allocation) + GuardBytes;
        check(hipMemset(allocation, Guard, bytes + 2 * GuardBytes), "initialize guards");
        check(hipStreamSynchronize(nullptr), "finish default-stream guard initialization");
    }
    ~Buffer() { if (allocation) { (void)hipDeviceSynchronize(); (void)hipFree(allocation); } }
    Buffer(const Buffer&) = delete;
    void upload(const void* host, hipStream_t stream) {
        check(hipMemcpyAsync(data, host, bytes, hipMemcpyHostToDevice, stream), "upload test bytes");
    }
    template <typename T> std::vector<T> read(hipStream_t stream) const {
        need(bytes % sizeof(T) == 0, "test read element size");
        std::vector<T> out(bytes / sizeof(T));
        check(hipMemcpyAsync(out.data(), data, bytes, hipMemcpyDeviceToHost, stream), "read test bytes");
        check(hipStreamSynchronize(stream), "finish test read");
        return out;
    }
    void guards(hipStream_t stream) const {
        std::array<std::uint8_t, GuardBytes> prefix{}, suffix{};
        check(hipMemcpyAsync(prefix.data(), allocation, GuardBytes, hipMemcpyDeviceToHost, stream), "prefix guard");
        check(hipMemcpyAsync(suffix.data(), static_cast<const std::uint8_t*>(data) + bytes,
                             GuardBytes, hipMemcpyDeviceToHost, stream), "suffix guard");
        check(hipStreamSynchronize(stream), "finish guard read");
        need(std::all_of(prefix.begin(), prefix.end(), [](auto x) { return x == Guard; }) &&
             std::all_of(suffix.begin(), suffix.end(), [](auto x) { return x == Guard; }), "guard was modified");
    }
    void* allocation = nullptr;
    void* data = nullptr;
    std::size_t bytes = 0;
};

class Graph {
public:
    ~Graph() {
        (void)hipDeviceSynchronize();
        if (exec) (void)hipGraphExecDestroy(exec);
        if (graph) (void)hipGraphDestroy(graph);
    }
    Graph() = default;
    Graph(const Graph&) = delete;
    template <typename Callable> void capture(hipStream_t stream, Callable&& body) {
        check(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal), "begin real math capture");
        body();
        check(hipStreamEndCapture(stream, &graph), "end real math capture");
        check(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate real math graph");
    }
    void replay(hipStream_t stream) {
        check(hipGraphLaunch(exec, stream), "replay same real math graph");
        check(hipStreamSynchronize(stream), "finish same graph replay");
    }
    hipGraph_t graph = nullptr;
    hipGraphExec_t exec = nullptr;
};

std::uint16_t bf16(float value) {
    // Independent CPU BF16 RNE; patterned values happen to be exactly representable.
    auto bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<std::uint16_t>(bits >> 16);
}
double fp64(std::uint16_t value) {
    return static_cast<double>(std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16));
}
std::vector<std::uint16_t> pattern(std::size_t count, unsigned seed, int denominator) {
    std::vector<std::uint16_t> out(count);
    for (std::size_t i = 0; i < count; ++i) {
        const auto mixed = static_cast<std::uint32_t>((i * 37u + (i / 17u) * 13u + seed * 23u) % 97u);
        out[i] = bf16(static_cast<float>(static_cast<int>(mixed) - 48) / denominator);
    }
    return out;
}
std::size_t logical_index(int heads, int h, int token, int d) {
    return static_cast<std::size_t>(d) + D * (h + static_cast<std::size_t>(heads) * token);
}

std::vector<double> oracle(const std::vector<std::uint16_t>& q,
                          const std::vector<std::uint16_t>& keys,
                          const std::vector<std::uint16_t>& values,
                          int kv_heads, int base, int width) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const int q_heads = kv_heads == 4 ? 24 : 16;
    const int group = q_heads / kv_heads;
    std::vector<double> out(static_cast<std::size_t>(D) * q_heads * width);
    std::vector<double> scores(base + width);
    for (int t = 0; t < width; ++t) for (int h = 0; h < q_heads; ++h) {
        const int kv = h / group, end = base + t + 1;
        double maximum = -std::numeric_limits<double>::infinity();
        for (int key = 0; key < end; ++key) {
            double dot = 0;
            for (int d = 0; d < D; ++d)
                dot += fp64(q[logical_index(q_heads, h, t, d)]) * fp64(keys[logical_index(kv_heads, kv, key, d)]);
            scores[key] = dot * static_cast<double>(Scale);
            maximum = std::max(maximum, scores[key]);
        }
        double denominator = 0;
        for (int key = 0; key < end; ++key) { scores[key] = std::exp(scores[key] - maximum); denominator += scores[key]; }
        for (int d = 0; d < D; ++d) {
            double numerator = 0;
            for (int key = 0; key < end; ++key)
                numerator += scores[key] * fp64(values[logical_index(kv_heads, kv, key, d)]);
            out[logical_index(q_heads, h, t, d)] = numerator / denominator;
        }
    }
    return out;
}

__global__ void poison_borrowed_span(std::uint8_t* bytes, std::uint64_t size) {
    for (std::uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) bytes[i] = 0xd3;
}

enum class Mode { AppendThenSmallCached, SmallInlineAppend, PromptAppend };

void one_case(int kv_heads, Mode mode, std::uint64_t& checks, std::uint64_t& replay_count) {
    const int q_heads = kv_heads == 4 ? 24 : 16;
    const int width = mode == Mode::PromptAppend ? 35 : 3;
    const int base = mode == Mode::AppendThenSmallCached ? 63 : (mode == Mode::SmallInlineAppend ? 127 : 95);
    const std::uint64_t slot_bytes = std::uint64_t{D} * 64 * kv_heads * 2;
    const std::size_t slot_words = static_cast<std::size_t>(slot_bytes / 2);
    LayoutBuilder layout_builder;
    DeviceKVPagePoolSpec spec;
    spec.page_group_count = Groups;
    spec.geometry.device_plane_order = PagedKVPlaneOrder::PageMajor;
    spec.geometry.planes = {{DType::BF16, D, kv_heads, 256}, {DType::BF16, D, kv_heads, 256}};
    const auto layout = plan_device_kv_page_pool(layout_builder, spec);
    KVSegmentedStorage::Config config;
    config.device = 0; config.planes = 2; config.page_groups = Groups; config.slot_bytes = slot_bytes;
    config.window_offset = 0; config.max_window_bytes = 4 * slot_bytes; config.max_user_streams = 2;
    KVSegmentedNativePool native(config, layout);
    auto& pool = native.page_pool();
    auto& storage = native.storage();
    Stream consumer, borrower;
    auto reservation = pool.reserve(LogicalPages);
    need(reservation.has_value(), "original allocator reserve refused");
    std::vector<DeviceKVPageLease> pages;
    pages.reserve(LogicalPages);
    pool.materialize(*reservation, LogicalPages, pages);
    need(pages.size() == LogicalPages, "original allocator materialize width");
    // Publish real minted handles through the ORIGINAL execution-table API, not
    // private index access or a guessed allocation sequence.
    LayoutBuilder table_builder;
    const auto table_layout = plan_kv_execution_tables(table_builder, {LogicalPages, 1});
    Buffer blocks(table_builder.finish());
    KVExecutionTablePool execution_tables(DeviceSpan{blocks.data, blocks.bytes}, table_layout, pool);
    auto execution_row = execution_tables.acquire(0);
    const std::array<DeviceKVPageHandle, LogicalPages> ordered_handles{
        pages[2].handle(), pages[0].handle(), pages[1].handle()};
    execution_tables.publish(execution_row.handle(), 0, ordered_handles, RankStreams{consumer.value});
    const auto table_vector = blocks.read<std::int32_t>(consumer.value);
    need(table_vector.size() == LogicalPages, "execution table has unexpected padding");
    const std::array<std::int32_t, LogicalPages> table{table_vector[0], table_vector[1], table_vector[2]};
    need(table[0] != table[1] && table[1] != table[2] && table[0] > table[1], "block table is not nonmonotonic");
    auto key_values = pattern(static_cast<std::size_t>(D) * kv_heads * LogicalTokens, 5, 128);
    auto value_values = pattern(key_values.size(), 7, 32);
    auto query_values = pattern(static_cast<std::size_t>(D) * q_heads * width, 11, 128);
    auto new_keys = pattern(static_cast<std::size_t>(D) * kv_heads * width, 17, 128);
    auto new_values = pattern(new_keys.size(), 19, 32);
    std::vector<std::int32_t> positions(width);
    for (int t = 0; t < width; ++t) positions[t] = base + t;
    const auto kp = pool.plane_view(0), vp = pool.plane_view(1);
    PagedKVLayerView cache;
    cache.k_pages = kp.storage; cache.v_pages = vp.storage;
    cache.block_table = execution_tables.row(execution_row.handle());
    cache.head_dim = D; cache.num_kv_heads = kv_heads; cache.storage = KvCacheStorage::BFloat16;
    cache.k_page_bases = kp.page_bases; cache.v_page_bases = vp.page_bases;
    need(cache.k_page_bases && cache.v_page_bases, "actual bound pool did not expose the stable tables");
    // Bound storage intentionally seals its raw initialization span. Initialize
    // live cells with the ACTUAL append body instead of reopening an affine alias.
    {
        Buffer initial_k(key_values.size() * 2), initial_v(value_values.size() * 2);
        std::vector<std::int32_t> initial_positions(LogicalTokens);
        for (int t = 0; t < LogicalTokens; ++t) initial_positions[t] = t;
        Buffer initial_pos(initial_positions.size() * sizeof(std::int32_t));
        initial_k.upload(key_values.data(), consumer.value); initial_v.upload(value_values.data(), consumer.value);
        initial_pos.upload(initial_positions.data(), consumer.value);
        Tensor ik(initial_k.data, DType::BF16, {D, kv_heads, LogicalTokens, 1});
        Tensor iv(initial_v.data, DType::BF16, {D, kv_heads, LogicalTokens, 1});
        Tensor ip(initial_pos.data, DType::I32, {LogicalTokens, 1, 1, 1});
        segmented_bf16_append(ik, iv, ip, cache, PagedKVPlaneOrder::PageMajor, consumer.value);
        check(hipStreamSynchronize(consumer.value), "finish initial real append");
        need(initial_k.read<std::uint16_t>(consumer.value) == key_values &&
             initial_v.read<std::uint16_t>(consumer.value) == value_values &&
             initial_pos.read<std::int32_t>(consumer.value) == initial_positions,
             "initial append modified its inputs");
        initial_k.guards(consumer.value); initial_v.guards(consumer.value); initial_pos.guards(consumer.value);
    }
    for (int t = 0; t < width; ++t) for (int h = 0; h < kv_heads; ++h) for (int d = 0; d < D; ++d) {
        key_values[logical_index(kv_heads, h, base + t, d)] = new_keys[logical_index(kv_heads, h, t, d)];
        value_values[logical_index(kv_heads, h, base + t, d)] = new_values[logical_index(kv_heads, h, t, d)];
    }
    const auto reference = oracle(query_values, key_values, value_values, kv_heads, base, width);
    Buffer q(query_values.size() * 2), k(new_keys.size() * 2), v(new_values.size() * 2);
    Buffer pos(positions.size() * 4), out(reference.size() * 2);
    q.upload(query_values.data(), consumer.value); k.upload(new_keys.data(), consumer.value);
    v.upload(new_values.data(), consumer.value); pos.upload(positions.data(), consumer.value);
    Tensor tq(q.data, DType::BF16, {D, q_heads, width, 1});
    Tensor tk(k.data, DType::BF16, {D, kv_heads, width, 1}), tv(v.data, DType::BF16, {D, kv_heads, width, 1});
    Tensor tp(pos.data, DType::I32, {width, 1, 1, 1}), to(out.data, DType::BF16, {D, q_heads, width, 1});
    const int max_splits = kv_heads == 4 ? 85 : 170;
    const auto stats = mode == Mode::PromptAppend ? std::uint64_t{1} :
        static_cast<std::uint64_t>(q_heads) * width * max_splits;
    Buffer acc(stats * D * sizeof(float)), maximum(stats * sizeof(float)), denominator(stats * sizeof(float));
    SegmentedBF16Partials partials{static_cast<float*>(acc.data), static_cast<float*>(maximum.data),
        static_cast<float*>(denominator.data), stats * D, stats};
    check(hipStreamSynchronize(consumer.value), "finish initial upload before capture");

    std::vector<std::uint64_t> before(2 * Groups), after(2 * Groups);
    check(hipMemcpyAsync(before.data(), cache.k_page_bases, before.size() * sizeof(std::uint64_t),
                         hipMemcpyDeviceToHost, consumer.value), "read original mapping");
    check(hipStreamSynchronize(consumer.value), "finish original mapping read");
    if (mode == Mode::PromptAppend) segmented_bf16_prepare_prompt(kv_heads, consumer.value);
    const auto launch_math = [&](const PagedKVLayerView& selected_cache) {
        if (mode == Mode::AppendThenSmallCached) {
            segmented_bf16_append(tk, tv, tp, selected_cache, PagedKVPlaneOrder::PageMajor, consumer.value);
            segmented_bf16_small_cached(tq, tp, Scale, selected_cache, PagedKVPlaneOrder::PageMajor,
                                        partials, to, consumer.value);
        } else if (mode == Mode::SmallInlineAppend) {
            segmented_bf16_small_append(tq, tk, tv, tp, Scale, selected_cache, PagedKVPlaneOrder::PageMajor,
                                        partials, to, consumer.value);
        } else {
            segmented_bf16_prompt_append(tq, tk, tv, tp, Scale, selected_cache,
                                         PagedKVPlaneOrder::PageMajor, to, consumer.value);
        }
    };
    // Independent EMPTY-table affine control runs only while the placement is
    // still identity. It is never replayed after movement; the persistent graph
    // below captures the stable table allocation instead.
    auto affine_cache = cache;
    affine_cache.k_page_bases = nullptr; affine_cache.v_page_bases = nullptr;
    launch_math(affine_cache);
    check(hipStreamSynchronize(consumer.value), "finish empty-table affine control");
    std::vector<std::uint16_t> first_output = out.read<std::uint16_t>(consumer.value);
    Graph graph;
    graph.capture(consumer.value, [&] { launch_math(cache); });
    const auto captured = graph.exec;
    KVSegmentedSpan borrowed_span{};
    const auto seed_fresh_destinations = [&] {
        // Real Core H2D follows the CURRENT map. Whole canonical Host page
        // records preserve every non-target history byte; only the new-token
        // rows are replaced with BF16 NaN. A captured no-op/wrong affine append
        // must therefore fail the exact cache oracle after EACH replay.
        const auto host_layout = plan_host_kv_page_layout(spec.geometry);
        const int first_page = base / 64, last_page = (base + width - 1) / 64;
        const auto touched = static_cast<std::size_t>(last_page - first_page + 1);
        PinnedHostBuffer records(host_layout.page_stride * touched);
        std::vector<const std::byte*> record_addresses(touched);
        std::vector<std::uint32_t> record_groups(touched, 0);
        std::vector<DeviceKVPageHandle> destinations;
        destinations.reserve(touched);
        for (std::size_t r = 0; r < touched; ++r) {
            const int logical_page = first_page + static_cast<int>(r);
            auto* record = static_cast<std::byte*>(records.data()) + r * host_layout.page_stride;
            record_addresses[r] = record;
            destinations.push_back(ordered_handles[logical_page]);
            for (int plane = 0; plane < 2; ++plane) {
                auto* page = reinterpret_cast<std::uint16_t*>(record + host_layout.planes[plane].offset);
                const auto& expected = plane == 0 ? key_values : value_values;
                for (int h = 0; h < kv_heads; ++h) for (int t = 0; t < 64; ++t) for (int d = 0; d < D; ++d) {
                    const int token = logical_page * 64 + t;
                    page[d + D * (t + 64 * h)] = token >= base && token < base + width ?
                        std::uint16_t{0xffff} : expected[logical_index(kv_heads, h, token, d)];
                }
            }
        }
        pool.copy_from_host_records(record_addresses, record_groups, destinations, host_layout,
                                     RankStreams{consumer.value});
        check(hipMemsetAsync(out.data, 0xff, out.bytes, consumer.value), "poison fresh output payload only");
        check(hipStreamSynchronize(consumer.value), "finish fresh mapped Core H2D/output poison");
        std::vector<std::uint64_t> current(2 * Groups);
        check(hipMemcpyAsync(current.data(), cache.k_page_bases, current.size() * 8,
                             hipMemcpyDeviceToHost, consumer.value), "read seed proof current table");
        check(hipStreamSynchronize(consumer.value), "finish seed table proof");
        std::array<std::uint16_t, D> row{};
        for (int plane = 0; plane < 2; ++plane) for (int t = 0; t < width; ++t) for (int h = 0; h < kv_heads; ++h) {
            const int token = base + t, group = table[token / 64];
            const auto* page = reinterpret_cast<const std::uint16_t*>(static_cast<std::uintptr_t>(current[plane * Groups + group]));
            const auto* source = page + D * (token % 64 + 64 * h);
            check(hipMemcpyAsync(row.data(), source, sizeof(row), hipMemcpyDeviceToHost, consumer.value), "read poisoned append row");
            check(hipStreamSynchronize(consumer.value), "finish poisoned append row proof");
            need(std::all_of(row.begin(), row.end(), [](auto x) { return x == 0xffff; }),
                 "pre-replay Core H2D did not poison every mapped target row");
        }
        q.guards(consumer.value); k.guards(consumer.value); v.guards(consumer.value);
        pos.guards(consumer.value); blocks.guards(consumer.value); out.guards(consumer.value);
    };
    const auto verify = [&] {
        need(graph.exec == captured && cache.k_page_bases == kp.page_bases &&
             cache.v_page_bases == vp.page_bases, "captured graph or device table allocation changed");
        const auto bits = out.read<std::uint16_t>(consumer.value);
        std::vector<double> actual(bits.size());
        std::transform(bits.begin(), bits.end(), actual.begin(), fp64);
        const auto result = ninfer::test::compute_reduction_stats(actual.data(), reference.data(), actual.size());
        // EXACT original BF16 Attention criterion, not a looser relocation bound.
        const ninfer::test::ReductionCriterion criterion{2.8e-3, 1.0e-3, ninfer::test::kBf16GrossRelativeFloor};
        need(ninfer::test::reduction_passes(result, actual.size(), criterion), "independent FP64 Attention oracle failed");
        need(bits == first_output, "table Graph differs from affine control or changed across relocation");
        need(q.read<std::uint16_t>(consumer.value) == query_values &&
             k.read<std::uint16_t>(consumer.value) == new_keys && v.read<std::uint16_t>(consumer.value) == new_values &&
             pos.read<std::int32_t>(consumer.value) == positions, "input was modified");
        const auto got_table = blocks.read<std::int32_t>(consumer.value);
        need(std::equal(got_table.begin(), got_table.end(), table.begin(), table.end()), "block table was modified");
        std::vector<std::uint64_t> mapping(2 * Groups);
        check(hipMemcpyAsync(mapping.data(), cache.k_page_bases, mapping.size() * 8,
                             hipMemcpyDeviceToHost, consumer.value), "read stable mapping");
        check(hipStreamSynchronize(consumer.value), "finish mapping read");
        std::vector<std::uint16_t> page(slot_words);
        for (int plane = 0; plane < 2; ++plane) for (int logical = 0; logical < LogicalPages; ++logical) {
            check(hipMemcpyAsync(page.data(), reinterpret_cast<const void*>(static_cast<std::uintptr_t>(
                mapping[plane * Groups + table[logical]])), slot_bytes, hipMemcpyDeviceToHost,
                consumer.value), "read true live BF16 cell");
            check(hipStreamSynchronize(consumer.value), "finish live cell read");
            const auto& expected = plane == 0 ? key_values : value_values;
            for (int h = 0; h < kv_heads; ++h) for (int token = 0; token < 64; ++token) for (int d = 0; d < D; ++d) {
                need(page[d + D * (token + 64 * h)] == expected[logical_index(kv_heads, h, logical * 64 + token, d)],
                     "actual append/cache bytes differ from independent BF16 scatter");
            }
        }
        for (const Buffer* buffer : {&q, &k, &v, &pos, &blocks, &out, &acc, &maximum, &denominator})
            buffer->guards(consumer.value);
        if (borrowed_span.data) {
            std::vector<std::uint8_t> bytes(static_cast<std::size_t>(borrowed_span.bytes));
            check(hipMemcpyAsync(bytes.data(), borrowed_span.data, bytes.size(), hipMemcpyDeviceToHost,
                                 consumer.value), "read active borrowed span preservation");
            check(hipStreamSynchronize(consumer.value), "finish active span read");
            need(std::all_of(bytes.begin(), bytes.end(), [](auto x) { return x == 0xd3; }),
                 "mapped Core transfer or captured append modified the active borrowed span");
        }
        checks += bits.size();
    };
    seed_fresh_destinations();
    graph.replay(consumer.value); ++replay_count; verify();
    const std::array<hipStream_t, 2> users{consumer.value, borrower.value};
    auto loan = storage.begin_loan(3 * slot_bytes + 1, users);
    need(loan.open() && loan.copied_live_bytes() >= 3 * slot_bytes, "test did not move real live BF16 cells");
    loan.wait_ready(consumer.value); loan.wait_ready(borrower.value);
    check(hipMemcpyAsync(after.data(), cache.k_page_bases, after.size() * 8, hipMemcpyDeviceToHost,
                         consumer.value), "read relocated mapping");
    check(hipStreamSynchronize(consumer.value), "finish relocated mapping read");
    need(before != after, "mapping did not change");
    const auto old_v_plane = reinterpret_cast<std::uintptr_t>(vp.storage.data);
    bool cross_plane = false;
    for (int g = 0; g < LogicalPages; ++g) cross_plane |= after[g] >= old_v_plane;
    need(cross_plane, "test did not exercise a live cell moved across semantic plane storage");
    const auto leased = loan.leased();
    borrowed_span = leased;
    poison_borrowed_span<<<64, 256, 0, borrower.value>>>(static_cast<std::uint8_t*>(leased.data), leased.bytes);
    check(hipGetLastError(), "launch true raw-span borrower");
    check(hipStreamSynchronize(borrower.value), "finish deterministic borrower overwrite");
    seed_fresh_destinations();
    graph.replay(consumer.value); ++replay_count; verify();
    need(loan.close_after_streams(users), "borrower completion/allocator return failed");
    borrowed_span = {}; // No use of the borrower's capability after return.
    need(pool.lent_pages() == 0 && !storage.poisoned(), "real allocator did not return excluded IDs");
    seed_fresh_destinations();
    graph.replay(consumer.value); ++replay_count; verify();
}
} // namespace

int main(int argc, char** argv) {
    if (argc == 1) {
        std::cout << "{\"status\":\"PLAN_ONLY_NOT_QUALIFIED\",\"gpu_called\":false,"
                     "\"scope\":\"PageMajor D256 BF16; actual native append/small/prompt; same Graph before/during/after\"}\n";
        return 0;
    }
    if (argc != 2 || std::string_view(argv[1]) != "--run") return 2;
    try {
        const auto* pageable = std::getenv("NINFER_HOST_PAGEABLE");
        need(!pageable || pageable[0] == '\0' || pageable[0] == '0',
             "candidate requires the actual Core pinned Host-record allocation path");
        check(hipSetDevice(0), "set gfx1201 test device");
        hipDeviceProp_t properties{};
        check(hipGetDeviceProperties(&properties, 0), "query test device");
        need(std::string_view(properties.gcnArchName).starts_with("gfx1201"), "candidate is gfx1201 only");
        std::uint64_t checks = 0, replays = 0;
        for (int heads : {4, 2}) for (auto mode : {Mode::AppendThenSmallCached, Mode::SmallInlineAppend, Mode::PromptAppend})
            one_case(heads, mode, checks, replays);
        std::cout << "{\"status\":\"FUNCTIONAL_ONLY_PASS\",\"geometry_cases\":6,\"same_graph_replays\":"
                  << replays << ",\"output_checks\":" << checks
                  << ",\"independent_fp64\":true,\"exact_cache_bytes\":true,\"relocation_bit_parity\":true,"
                     "\"fresh_output_each_replay\":true,\"fresh_append_rows_each_replay\":true,"
                     "\"pending_ordering_witnessed\":false,\"concurrent_allocator_mutation_tested\":false,"
                     "\"production_gate\":false,\"engine_qualification\":false,\"performance_sample\":false}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
