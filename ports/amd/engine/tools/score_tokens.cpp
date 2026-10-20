// Research-only teacher-forced export through the unchanged public Engine.
// Default execution validates the input and prints a plan. Model/GPU execution
// requires --run and an external owner-held GPU lock/resource guard.
#include "ninfer/engine.h"
#include "nlohmann/json.hpp"

#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace {
using Json = nlohmann::json;
namespace fs = std::filesystem;

struct Options {
    bool run = false;
    bool help = false;
    bool prefill_a8 = false;
    std::uint32_t resource_hold_ms = 0;
    fs::path artifact, tokens, output;
    std::uint32_t first_target = 0;
    std::uint32_t context = 512;
    int device = 0;
    std::string kv_name = "bf16";
    ninfer::KvCacheStorage kv = ninfer::KvCacheStorage::BFloat16;
    std::string comparison_id, model_id, weights_id, tokenizer_id;
    std::string engine_label = "ninfer-amd-native";
};

constexpr std::string_view Usage =
    "usage: ninfer_amd_score_tokens --artifact MODEL.ninfer --tokens IDS.json\n"
    "       --first-target N --output SCORES.json [--context N] [--device N]\n"
    "       [--kv-dtype bf16|int8|fp8|rk8v4|rk4v4|rk4v4-e8|rk2v4-e8|nvfp4|k8v4]\n"
    "       [--comparison-id LABEL] [--model-id LABEL] [--weights-id LABEL]\n"
    "       [--tokenizer-id LABEL] [--engine-label LABEL] [--prefill-a8] [--run]\n"
    "       [--resource-hold-ms 0..10000] (quality resource observation only)\n"
    "       Defaults: context512, bf16 KV, device0, plan-only.\n"
    "\nIDS.json must be one JSON array of nonnegative int32 token IDs, with at\n"
    "least two IDs and at most --context IDs. first-target is a zero-based\n"
    "position in [1,token_count-1]. Every input ID is retained, including BOS,\n"
    "EOS and special IDs; no template, tokenizer, stop rule or sampling is used.\n"
    "Default: input/paths plan only, no Engine, no result file. --run must be\n"
    "launched under the external GPU lock and memory guard. Scores are natural\n"
    "log p(T[i]|T[0..i)), with fresh State/KV and the public-token normalizer.\n"
    "FP32 GDN state, native RoPE, no spec/graphs/cache/weight transcoding and\n"
    "device-profile off are fixed. A16 is the default; --prefill-a8 permits\n"
    "eligible integer activation routes and changes the quality profile. This\n"
    "permission is not an observed integer-kernel witness. Identity labels are declared\n"
    "comparison metadata; artifact identity and SHA256 belong in the sidecar.\n";

std::uint32_t unsigned_argument(std::string_view text, std::string_view label) {
    std::uint32_t value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || error != std::errc{} || end != text.data() + text.size()) {
        throw std::invalid_argument(std::string(label) + " requires a decimal uint32 value");
    }
    return value;
}

ninfer::KvCacheStorage kv_storage(std::string_view value) {
    using K = ninfer::KvCacheStorage;
    if (value == "bf16") return K::BFloat16;
    if (value == "int8") return K::Int8Group64;
    if (value == "fp8") return K::Fp8E4M3Row256;
    if (value == "rk8v4") return K::RotatedInt8KeyInt4ValueGroup64;
    if (value == "rk4v4") return K::RotatedLloyd4KeyInt4Value;
    if (value == "rk4v4-e8") return K::RotatedInt4KeyInt4ValueE8;
    if (value == "rk2v4-e8") return K::RotatedE8RootKeyInt4Value;
    if (value == "nvfp4") return K::Nvfp4Group16;
    if (value == "k8v4") return K::Fp8KeyNvfp4Value;
    throw std::invalid_argument("unsupported --kv-dtype");
}

Options parse_options(int argc, char** argv) {
    Options options;
    std::set<std::string_view> seen;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (!seen.insert(flag).second) {
            throw std::invalid_argument("duplicate option: " + std::string(flag));
        }
        if (flag == "--help") { options.help = true; continue; }
        if (flag == "--run") { options.run = true; continue; }
        if (flag == "--prefill-a8") { options.prefill_a8 = true; continue; }
        if (flag != "--artifact" && flag != "--tokens" && flag != "--output" &&
            flag != "--first-target" && flag != "--context" && flag != "--device" &&
            flag != "--kv-dtype" && flag != "--comparison-id" && flag != "--model-id" &&
            flag != "--weights-id" && flag != "--tokenizer-id" && flag != "--engine-label" &&
            flag != "--resource-hold-ms") {
            throw std::invalid_argument("unknown option: " + std::string(flag));
        }
        if (++i == argc) throw std::invalid_argument("missing value for " + std::string(flag));
        const std::string value = argv[i];
        if (value.empty()) throw std::invalid_argument("empty value for " + std::string(flag));
        if (flag == "--artifact") options.artifact = fs::u8path(value);
        else if (flag == "--tokens") options.tokens = fs::u8path(value);
        else if (flag == "--output") options.output = fs::u8path(value);
        else if (flag == "--first-target") options.first_target = unsigned_argument(value, flag);
        else if (flag == "--context") options.context = unsigned_argument(value, flag);
        else if (flag == "--resource-hold-ms") {
            options.resource_hold_ms = unsigned_argument(value, flag);
            if (options.resource_hold_ms > 10000)
                throw std::invalid_argument("--resource-hold-ms exceeds 10000");
        }
        else if (flag == "--device") {
            const auto device = unsigned_argument(value, flag);
            if (device > static_cast<unsigned>(std::numeric_limits<int>::max()))
                throw std::invalid_argument("--device exceeds int32");
            options.device = static_cast<int>(device);
        } else if (flag == "--kv-dtype") {
            options.kv_name = value;
            options.kv = kv_storage(value);
        } else if (flag == "--comparison-id") options.comparison_id = value;
        else if (flag == "--model-id") options.model_id = value;
        else if (flag == "--weights-id") options.weights_id = value;
        else if (flag == "--tokenizer-id") options.tokenizer_id = value;
        else if (flag == "--engine-label") options.engine_label = value;
    }
    if (options.help) return options;
    if (options.artifact.empty() || options.tokens.empty() || options.output.empty() ||
        !seen.contains("--first-target")) {
        throw std::invalid_argument("--artifact, --tokens, --first-target and --output are required");
    }
    if (options.context < 2) throw std::invalid_argument("--context must be at least two");
    options.artifact = fs::absolute(options.artifact).lexically_normal();
    options.tokens = fs::absolute(options.tokens).lexically_normal();
    options.output = fs::absolute(options.output).lexically_normal();
    if (!fs::is_regular_file(options.artifact)) throw std::invalid_argument("artifact is not a file");
    if (!fs::is_regular_file(options.tokens)) throw std::invalid_argument("token IDs are not a file");
    if (!fs::is_directory(options.output.parent_path()))
        throw std::invalid_argument("output parent directory must already exist");
    fs::path partial = options.output;
    partial += ".partial";
    if (fs::exists(options.output) || fs::exists(partial))
        throw std::invalid_argument("output or its .partial file already exists");
    if (options.output == options.tokens || options.output == options.artifact)
        throw std::invalid_argument("output must differ from both inputs");
    return options;
}

std::string path_utf8(const fs::path& path) {
    const auto bytes = path.generic_u8string();
    return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

std::vector<ninfer::TokenId> read_tokens(const Options& options) {
    // Bounded text acquisition, never an artifact/weight scan. A canonical
    // int32 array requires at most 12 bytes per item; allow extra whitespace.
    constexpr std::uint64_t maximum_file_bytes = 64ULL << 20;
    const auto size = fs::file_size(options.tokens);
    if (size > maximum_file_bytes || size > 32ULL * options.context + 4096ULL)
        throw std::invalid_argument("ID file exceeds the bounded context-sized JSON input limit");
    std::ifstream input(options.tokens, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open token ID file");
    const Json document = Json::parse(input); // Strict trailing data; comments are not accepted.
    if (!document.is_array() || document.size() < 2 || document.size() > options.context)
        throw std::invalid_argument("ID JSON must be an array with size in [2,context]");
    std::vector<ninfer::TokenId> tokens;
    tokens.reserve(document.size());
    for (std::size_t i = 0; i < document.size(); ++i) {
        const Json& item = document[i];
        if (!item.is_number_integer())
            throw std::invalid_argument("non-integer token ID at position " + std::to_string(i));
        const bool unsigned_number = item.is_number_unsigned();
        if ((!unsigned_number && item.get<std::int64_t>() < 0) ||
            (unsigned_number ? item.get<std::uint64_t>() >
                                   static_cast<std::uint64_t>(std::numeric_limits<ninfer::TokenId>::max())
                             : item.get<std::int64_t>() > std::numeric_limits<ninfer::TokenId>::max()))
            throw std::invalid_argument("token ID outside nonnegative int32 at position " +
                                        std::to_string(i));
        tokens.push_back(item.get<ninfer::TokenId>());
    }
    if (options.first_target == 0 || options.first_target >= tokens.size())
        throw std::invalid_argument("first-target must be in [1,token_count-1]");
    return tokens;
}

ninfer::EngineOptions engine_options(const Options& options) {
    ninfer::EngineOptions result;
    result.artifact_path = options.artifact;
    result.purpose = ninfer::EnginePurpose::CausalScoring;
    result.device = options.device;
    result.max_context = options.context;
    result.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(options.context);
    result.kv_cache = options.kv;
    result.gdn_state_fp16 = false;
    result.gdn_state_storage = ninfer::GdnStateStorage::FP32;
    result.mlp_a8_decode = false;
    result.prefill_a8 = options.prefill_a8;
    result.prefill_cublas = false;
    result.prefill_cublas_projections = false;
    result.fast_prefill_kernel = false;
    result.lm_head_q4 = false;
    result.lm_head_q6 = false;
    result.embedding_q4 = false;
    result.embedding_q6 = false;
    result.mtp_experts_q4 = false;
    result.quantize_output_head_fp8 = false;
    result.quantize_token_embedding_fp8 = false;
    result.speculative = {};
    result.use_cuda_graph = false;
    result.enable_vision = false;
    result.context_cache.enabled = false;
    result.device_profile = "off";
    return result;
}

Json execution_json(const ninfer::EngineOptions& options, std::string_view kv_name) {
    return {{"purpose", "CausalScoring"}, {"device", options.device},
            {"max_context", options.max_context}, {"kv_dtype", kv_name},
            {"kv_capacity_mode", "explicit"},
            {"kv_capacity_tokens", options.kv_capacity.explicit_tokens},
            {"prefill_chunk", options.prefill_chunk}, {"gdn_state", "fp32"},
            {"activation_policy", options.prefill_a8 ? "REQUESTED_ALLOW_A8_INT" : "A16"},
            {"prefill_a8", options.prefill_a8}, {"actual_integer_route_witnessed", false},
            {"mlp_a8_decode", options.mlp_a8_decode},
            {"prefill_cublas", options.prefill_cublas},
            {"prefill_cublas_projections", options.prefill_cublas_projections},
            {"fast_prefill_kernel", options.fast_prefill_kernel},
            {"speculative_backend", "none"}, {"use_cuda_graph", options.use_cuda_graph},
            {"context_cache", options.context_cache.enabled},
            {"enable_vision", options.enable_vision}, {"max_concurrency", options.max_concurrency},
            {"device_profile", options.device_profile}, {"rope_yarn", options.rope_yarn},
            {"rope_yarn_factor", options.rope_yarn_factor},
            {"rope_scaling_factor", options.rope_scaling_factor},
            {"rope_scaling_original_context", options.rope_scaling_original_context},
            {"load_time_weight_transcoding", false}};
}

void verify_effective_options(const Options& requested, const ninfer::EngineOptions& actual) {
    if (actual.purpose != ninfer::EnginePurpose::CausalScoring ||
        actual.device != requested.device || actual.max_context != requested.context ||
        actual.kv_cache != requested.kv || actual.kv_capacity.mode != ninfer::KvCapacityMode::Explicit ||
        actual.kv_capacity.explicit_tokens != requested.context || actual.prefill_chunk != 1024 ||
        actual.max_concurrency != 1 || actual.max_pending_requests != 1 || actual.gdn_state_fp16 ||
        actual.gdn_state_storage != ninfer::GdnStateStorage::FP32 ||
        actual.prefill_a8 != requested.prefill_a8 ||
        actual.mlp_a8_decode || actual.prefill_cublas || actual.prefill_cublas_projections ||
        actual.fast_prefill_kernel || actual.lm_head_q4 || actual.lm_head_q6 ||
        actual.embedding_q4 || actual.embedding_q6 || actual.mtp_experts_q4 ||
        actual.quantize_output_head_fp8 || actual.quantize_token_embedding_fp8 ||
        actual.speculative.backend != ninfer::SpeculativeBackend::None ||
        actual.speculative.draft_tokens != 0 || actual.speculative.lookup_ngram != 0 ||
        actual.speculative.ngram_draft_tokens != 0 || actual.use_cuda_graph ||
        actual.enable_vision || actual.context_cache.enabled || actual.device_profile != "off" ||
        actual.rope_yarn || actual.rope_yarn_factor != 1.0F ||
        actual.rope_scaling_factor != 1.0F || actual.rope_scaling_original_context != 0) {
        throw std::runtime_error("effective Engine options differ from the fixed scoring contract");
    }
}

bool public_token_valid(const ninfer::Engine& engine, ninfer::TokenId id) {
    try {
        (void)engine.token_bytes(id);
        return true;
    } catch (const std::out_of_range& error) {
        // Catch only the real frontend invalid-token contract. A different
        // out_of_range/runtime failure must not silently shrink the vocabulary.
        if (std::string_view(error.what()) == "token id is outside the tokenizer vocabulary")
            return false;
        throw;
    }
}

std::uint32_t public_token_count(const ninfer::Engine& engine, std::uint64_t physical_rows) {
    // Current model_metadata().vocab_size is assigned config.text.vocab_size in
    // model_instance.cpp, despite its comment saying tokenizer domain. Keep the
    // physical extent distinct from the actual public tokenizer boundary.
    if (physical_rows == 0 || physical_rows >
                                  static_cast<std::uint64_t>(std::numeric_limits<ninfer::TokenId>::max()))
        throw std::runtime_error("physical vocabulary cannot be bounded by public TokenId");
    if (!public_token_valid(engine, 0) ||
        public_token_valid(engine, static_cast<ninfer::TokenId>(physical_rows)))
        throw std::runtime_error("public token boundary violates the loaded contiguous-domain contract");
    std::uint32_t valid = 0, invalid = static_cast<std::uint32_t>(physical_rows);
    while (invalid - valid > 1) {
        const auto middle = valid + (invalid - valid) / 2;
        if (public_token_valid(engine, static_cast<ninfer::TokenId>(middle))) valid = middle;
        else invalid = middle;
    }
    return invalid;
}

Json common_json(const Options& options, const std::vector<ninfer::TokenId>& tokens) {
    return {{"schema_version", 1}, {"log_base", "e"}, {"token_ids", tokens},
            {"first_target", options.first_target}, {"token_count", tokens.size()},
            {"input_format", "json_integer_array"},
            {"input_ids_path", path_utf8(options.tokens)},
            {"artifact_path", path_utf8(options.artifact)},
            {"output_path", path_utf8(options.output)},
            {"resource_observation_hold_ms_requested", options.resource_hold_ms},
            {"input_semantics", {{"retokenized", false}, {"implicit_special_tokens", false},
                                 {"stop_truncation", false}, {"window_truncation", false},
                                 {"fresh_state_and_kv", true},
                                 {"position_indexing", "zero_based_input_array"}}},
            {"declared_identity", {{"comparison_id", options.comparison_id},
                                   {"model_id", options.model_id}, {"weights_id", options.weights_id},
                                   {"tokenizer_id", options.tokenizer_id},
                                   {"engine_label", options.engine_label},
                                   {"provenance", "caller_labels_unverified_by_cpp; sidecar owns artifact identity and SHA256"}}}};
}

void write_result(const fs::path& output, const Json& document) {
    fs::path temporary = output;
    temporary += ".partial";
    if (fs::exists(output) || fs::exists(temporary))
        throw std::runtime_error("output or its .partial file already exists");
    bool owns_temporary = false;
    try {
        std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
        if (!stream) throw std::runtime_error("cannot create result .partial file");
        owns_temporary = true;
        stream << document.dump(2) << '\n';
        stream.flush();
        if (!stream) throw std::runtime_error("cannot write complete result JSON");
        stream.close();
        if (!stream) throw std::runtime_error("cannot close complete result JSON");
        if (fs::exists(output)) throw std::runtime_error("output appeared while scoring");
        fs::rename(temporary, output);
        owns_temporary = false;
    } catch (...) {
        if (owns_temporary) {
            std::error_code ignored;
            fs::remove(temporary, ignored);
        }
        throw;
    }
}

int run(const Options& options) {
    const std::vector<ninfer::TokenId> tokens = read_tokens(options);
    Json result = common_json(options, tokens);
    if (!options.run) {
        result["status"] = "plan_only";
        result["engine_constructed"] = false;
        result["public_domain_checked"] = false;
        result["scored_tokens"] = tokens.size() - options.first_target;
        result["execution_requested"] = execution_json(engine_options(options), options.kv_name);
        result["normalization_note"] = "CausalScoring fixes actual prefill_chunk=1024 and max_concurrency=1; no model loaded";
        std::cout << result.dump(2) << '\n';
        return 0;
    }
    ninfer::Engine engine(engine_options(options));
    verify_effective_options(options, engine.options());
    const auto metadata = engine.model_metadata();
    const auto domain = public_token_count(engine, metadata.vocab_size);
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        if (static_cast<std::uint32_t>(tokens[i]) >= domain)
            throw std::invalid_argument("token ID outside public domain at position " + std::to_string(i));
    }
    // Sole model mathematical operation. No stop handling or token generation.
    const auto logprobs = engine.score_tokens(tokens, options.first_target);
    if (logprobs.size() != tokens.size() - options.first_target)
        throw std::runtime_error("Engine returned an invalid score count");
    Json scores = Json::array();
    double total_nll = 0.0;
    for (std::size_t i = 0; i < logprobs.size(); ++i) {
        const double logprob = logprobs[i];
        if (!std::isfinite(logprob) || logprob > 0.0)
            throw std::runtime_error("Engine returned non-finite or positive logprob at position " +
                                     std::to_string(options.first_target + i));
        const std::size_t position = options.first_target + i;
        scores.push_back({{"position", position}, {"target_id", tokens[position]}, {"logprob", logprob}});
        total_nll -= logprob;
    }
    if (!std::isfinite(total_nll)) throw std::runtime_error("non-finite aggregate NLL");
    const auto load = engine.load_summary();
    result["status"] = "computed";
    result["public_token_count"] = domain;
    result["physical_vocab_size"] = metadata.vocab_size;
    result["logprob_vocabulary"] = {{"first_token_id", 0}, {"token_count", domain}};
    result["domain_provenance"] = {
        {"public", "public Engine::token_bytes contiguous validity boundary; id0 valid and physical_rows invalid; loader enforces contiguity"},
        {"physical", "current ModelMetadata.vocab_size assigned config.text.vocab_size"},
        {"normalizer", "production target_logprobs receives model.resources.public_token_count; rows outside this prefix excluded"}};
    result["engine"] = {{"purpose", "CausalScoring"},
                         {"identity_provenance", "unchanged public Engine reporting; not verified artifact-header identity"},
                         {"load_summary", {{"architecture", load.architecture},
                                           {"model_name", load.model_name},
                                           {"prefill_signature", load.prefill_signature},
                                           {"weight_formats", load.weight_formats}}},
                         {"model_metadata", {{"model_id", metadata.model_id},
                                             {"weights_id", metadata.weights_id},
                                             {"vocab_size", metadata.vocab_size},
                                             {"embedding_size", metadata.embedding_size},
                                             {"native_context", metadata.native_context},
                                             {"parameters", metadata.parameters},
                                             {"weight_bytes", metadata.weight_bytes}}}};
    result["execution"] = execution_json(engine.options(), options.kv_name);
    result["scores"] = std::move(scores);
    result["scored_tokens"] = logprobs.size();
    result["total_nll"] = total_nll;
    result["mean_nll"] = total_nll / static_cast<double>(logprobs.size());
    // Keep the real Engine alive for a slow external Windows counter reader.
    // This explicit research-only wait is outside scoring, never a timing sample.
    const auto hold_begin = std::chrono::steady_clock::now();
    if (options.resource_hold_ms != 0)
        std::this_thread::sleep_for(std::chrono::milliseconds(options.resource_hold_ms));
    result["resource_observation"] = {
        {"requested_hold_ms", options.resource_hold_ms},
        {"actual_hold_ms", std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - hold_begin).count()},
        {"engine_alive_during_hold", true}, {"performance_sample", false}};
    write_result(options.output, result);
    std::cerr << "teacher-forced scores computed: " << logprobs.size()
              << " natural-log values; no quality/performance qualification implied\n";
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (options.help) { std::cout << Usage; return 0; }
        return run(options);
    } catch (const std::exception& error) {
        std::cerr << Json{{"schema_version", 1}, {"status", "error"},
                          {"message", error.what()}}.dump() << '\n';
        return 1;
    }
}
