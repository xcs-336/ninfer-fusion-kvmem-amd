// Research-only CPU prompt export. This translation unit intentionally includes
// the generated production frontend implementation so its private rendering
// helpers are reused without changing the production API or copying a template.
// Do not also add frontend.cpp as a source of this executable.
#include "models/qwen3_5/frontend/frontend.cpp"

#include "product/prompt_input/prompt_input.h"

#include <charconv>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <system_error>

namespace frontend_export_tool {

namespace qwen = ninfer::models::qwen3_5;
namespace fi = ninfer::models::qwen3_5::frontend;
using Json = nlohmann::ordered_json;

constexpr std::size_t kMiB = 1024U * 1024U;
constexpr std::uint32_t kMaximumExportContext = 1024U * 1024U;

struct Options {
    std::filesystem::path resources;
    std::filesystem::path messages;
    std::filesystem::path prompt_file;
    std::filesystem::path output;
    std::optional<std::string> prompt;
    // Match the explicit --no-thinking used by the initial native Bonsai run.
    // "default" can instead preserve the actual template's unset option.
    std::optional<bool> thinking = false;
    std::uint32_t max_context = 8192;
    ninfer::StopPolicy stop;
};

std::string path_utf8(const std::filesystem::path& path) {
    const auto value = path.generic_u8string();
    return std::string(value.begin(), value.end());
}

std::uint32_t parse_u32(std::string_view text, std::uint32_t maximum,
                        std::string_view option, bool allow_zero = false) {
    std::uint32_t value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
        (!allow_zero && value == 0) || value > maximum) {
        throw std::invalid_argument(std::string(option) + " has an invalid integer value");
    }
    return value;
}

void usage(std::ostream& stream) {
    stream <<
        "CPU-only production frontend export (research schema 1)\n"
        "  --resources DIR  config.json, tokenizer.json, tokenizer_config.json,\n"
        "                   generation_config.json, chat_template.jinja only\n"
        "  exactly one of --messages FILE | --prompt TEXT | --prompt-file FILE\n"
        "  --thinking off|on|default  default: off (native smoke configuration)\n"
        "  --max-context N            default: 8192, maximum: 1048576\n"
        "  --stop-token-id N          repeat to augment actual model defaults\n"
        "  --output FILE             default: JSON to stdout\n"
        "Pure text, new assistant turn only; no tools, media, or explicit cache markers.\n"
        "UTF-8 JSON/text files preserve Unicode independently of the shell code page.\n"
        "This program never opens a .ninfer artifact or constructs Engine/DeviceContext.\n";
}

Options parse_options(int argc, char** argv) {
    Options result;
    std::vector<std::string> single_options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view option(argv[i]);
        if (option == "--help" || option == "-h") {
            usage(std::cout);
            std::exit(0);
        }
        if (i + 1 >= argc) {
            throw std::invalid_argument("missing value for " + std::string(option));
        }
        const std::string_view value(argv[++i]);
        if (option != "--stop-token-id") {
            if (std::find(single_options.begin(), single_options.end(), option) !=
                single_options.end()) {
                throw std::invalid_argument("duplicate option " + std::string(option));
            }
            single_options.emplace_back(option);
        }
        if (option == "--resources") {
            result.resources = std::filesystem::u8path(value);
        } else if (option == "--messages") {
            result.messages = std::filesystem::u8path(value);
        } else if (option == "--prompt-file") {
            result.prompt_file = std::filesystem::u8path(value);
        } else if (option == "--prompt") {
            result.prompt = std::string(value);
        } else if (option == "--output") {
            result.output = std::filesystem::u8path(value);
        } else if (option == "--thinking") {
            if (value == "off") { result.thinking = false; }
            else if (value == "on") { result.thinking = true; }
            else if (value == "default") { result.thinking = std::nullopt; }
            else { throw std::invalid_argument("--thinking requires off, on, or default"); }
        } else if (option == "--max-context") {
            result.max_context = parse_u32(value, kMaximumExportContext, option);
        } else if (option == "--stop-token-id") {
            result.stop.token_ids.push_back(static_cast<ninfer::TokenId>(parse_u32(
                value, static_cast<std::uint32_t>(std::numeric_limits<ninfer::TokenId>::max()),
                option, true)));
        } else {
            throw std::invalid_argument("unknown option " + std::string(option));
        }
    }
    const unsigned inputs = static_cast<unsigned>(!result.messages.empty()) +
                            static_cast<unsigned>(!result.prompt_file.empty()) +
                            static_cast<unsigned>(result.prompt.has_value());
    if (result.resources.empty() || inputs != 1) {
        throw std::invalid_argument("--resources and exactly one prompt input are required");
    }
    if (result.prompt && result.prompt->size() > 2 * kMiB) {
        throw std::invalid_argument("--prompt exceeds the 2 MiB research input bound");
    }
    return result;
}

std::uintmax_t bounded_file_size(const std::filesystem::path& path, std::size_t maximum) {
    if (!std::filesystem::is_regular_file(path)) {
        throw std::invalid_argument("not a regular input file: " + path_utf8(path));
    }
    const auto size = std::filesystem::file_size(path);
    if (size == 0 || size > maximum) {
        throw std::invalid_argument("input is empty or exceeds its research byte bound: " +
                                    path_utf8(path));
    }
    return size;
}

std::string read_bounded(const std::filesystem::path& path, std::size_t maximum) {
    const auto size = bounded_file_size(path, maximum);
    std::ifstream stream(path, std::ios::binary);
    if (!stream) { throw std::runtime_error("cannot read " + path_utf8(path)); }
    std::string result(static_cast<std::size_t>(size), '\0');
    if (!stream.read(result.data(), static_cast<std::streamsize>(result.size())) ||
        stream.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("input changed or could not be read: " + path_utf8(path));
    }
    return result;
}

struct ResourceStorage {
    std::string config;
    std::string tokenizer;
    std::string tokenizer_config;
    std::string generation_config;
    std::string chat_template;
};

ResourceStorage read_resources(const std::filesystem::path& directory) {
    // These exact five small files are the entire resource I/O surface. In
    // particular, there is no artifact reader, directory scan, or model loader.
    return {
        .config = read_bounded(directory / "config.json", kMiB),
        .tokenizer = read_bounded(directory / "tokenizer.json", 32 * kMiB),
        .tokenizer_config = read_bounded(directory / "tokenizer_config.json", 2 * kMiB),
        .generation_config = read_bounded(directory / "generation_config.json", kMiB),
        .chat_template = read_bounded(directory / "chat_template.jinja", 2 * kMiB),
    };
}

std::uint32_t physical_vocab_size(const std::string& config_bytes) {
    const Json config = Json::parse(config_bytes);
    if (!config.is_object() || !config.contains("text_config") ||
        !config.at("text_config").is_object() ||
        !config.at("text_config").contains("vocab_size")) {
        throw std::invalid_argument("expected Qwen3.5 config.text_config.vocab_size");
    }
    const Json& value = config.at("text_config").at("vocab_size");
    if (!value.is_number_unsigned()) {
        throw std::invalid_argument("config.text_config.vocab_size must be a positive integer");
    }
    const auto count = value.get<std::uint64_t>();
    if (!count || count > static_cast<std::uint64_t>(std::numeric_limits<ninfer::TokenId>::max())) {
        throw std::invalid_argument("physical vocabulary exceeds the native token-id domain");
    }
    return static_cast<std::uint32_t>(count);
}

ninfer::PromptInput read_prompt(const Options& options) {
    if (!options.messages.empty()) {
        // Bound the file before invoking the unchanged product JSON parser. It
        // rejects media with vision=false before acquisition/network code runs.
        (void)bounded_file_size(options.messages, 2 * kMiB);
        return ninfer::product::prompt_from_messages(options.messages, options.thinking, false);
    }
    const std::string text = options.prompt ? *options.prompt :
                            read_bounded(options.prompt_file, 2 * kMiB);
    (void)ninfer::text::unicode_internal::utf8_codepoints(text, "prompt input");
    return ninfer::product::prompt_from_text(text, options.thinking);
}

void require_replayable_text(const ninfer::PromptInput& input) {
    // Frontend::prepare appends a ToolBoundary marker when tools are declared.
    // That mode is deliberately not reconstructed by this bounded first tool.
    if (!input.options.tool_jsons.empty() || !input.options.forced_tool_name.empty() ||
        !input.context_cache.markers.empty()) {
        throw std::invalid_argument("export supports text without tools or explicit cache markers");
    }
    if (input.options.continuation != ninfer::PromptContinuationMode::NewAssistantTurn ||
        input.options.reasoning_effort || input.options.preserve_thinking ||
        !input.options.chat_template_kwargs_json.empty() || input.options.add_vision_id) {
        throw std::invalid_argument("unsupported prompt option in bounded frontend export");
    }
    for (const auto& message : input.messages) {
        if (message.role == ninfer::ChatRole::Tool || !message.tool_calls.empty() ||
            !message.tool_call_id.empty()) {
            throw std::invalid_argument("tool-role/tool-call history is outside this export scope");
        }
        for (const auto& part : message.parts) {
            if (part.kind != ninfer::MessagePartKind::Text) {
                throw std::invalid_argument("media is outside this CPU text export scope");
            }
        }
    }
}

std::string hex_bytes(std::string_view value) {
    constexpr char digits[] = "0123456789abcdef";
    std::string result(value.size() * 2U, '\0');
    for (std::size_t i = 0; i < value.size(); ++i) {
        const auto byte = static_cast<unsigned char>(value[i]);
        result[2 * i] = digits[byte >> 4U];
        result[2 * i + 1] = digits[byte & 15U];
    }
    return result;
}

Json token_piece(const fi::Tokenizer& tokenizer, ninfer::TokenId id) {
    const fi::DecodedTokenView decoded = tokenizer.decoded_token(id);
    Json result{{"id", id}, {"bytes_hex", hex_bytes(decoded.bytes)}, {"special", decoded.special}};
    // A BPE token can contain a partial UTF-8 codepoint. Never insert those bytes
    // into a JSON string or replace them with U+FFFD; hex remains authoritative.
    try {
        (void)ninfer::text::unicode_internal::utf8_codepoints(decoded.bytes, "token bytes");
        result["utf8"] = std::string(decoded.bytes);
    } catch (const std::invalid_argument&) {
        result["utf8"] = nullptr;
    }
    return result;
}

Json export_prompt(const Options& options) {
    const ResourceStorage storage = read_resources(options.resources);
    const std::uint32_t physical_count = physical_vocab_size(storage.config);
    qwen::FrontendResources resources{
        .tokenizer_json = storage.tokenizer,
        .tokenizer_config_json = storage.tokenizer_config,
        .chat_template_jinja = storage.chat_template,
        .generation_config_json = storage.generation_config,
    };
    // Use the actual loader's tokenizer/public-domain validation. Only its text
    // row bound is needed; vision and draft validation belong to a model load,
    // which this tool neither performs nor claims to validate.
    qwen::Config text_bounds;
    text_bounds.text.vocab_size = physical_count;
    qwen::parse_resources(resources, text_bounds);

    qwen::FrontendOptions frontend_options;
    frontend_options.vision_enabled = false;
    frontend_options.max_context = options.max_context;
    const qwen::Frontend frontend = qwen::make_frontend(resources, frontend_options);
    const ninfer::PromptInput input = read_prompt(options);
    require_replayable_text(input);

    // Authority: the original public preparation route and its original test
    // access. The retained input is an exact copy of what prepare receives.
    const qwen::PreparedPrompt prepared = frontend.prepare(input);
    const qwen::PreparedPromptData& actual = qwen::FrontendTestAccess::inspect(prepared);
    const auto messages = qwen::convert_messages(input.messages);
    const auto compiled = qwen::compile_chat_template(resources, frontend_options.chat_template_path);
    const fi::RenderedChat rendered = compiled.render(
        messages, qwen::render_options(input.options, input.context_cache.markers));
    const fi::EncodedChat encoded = fi::encode_rendered_chat(
        *resources.tokenizer, rendered, static_cast<std::size_t>(options.max_context) + 1U);
    if (encoded.input_ids != actual.token_ids ||
        rendered.starts_in_reasoning != actual.starts_in_reasoning) {
        throw std::logic_error("production rendered replay disagrees with Frontend::prepare");
    }
    (void)ninfer::text::unicode_internal::utf8_codepoints(rendered.text, "rendered prompt");
    const ninfer::StopPolicy effective = qwen::merge_stop_policy(*resources.tokenizer, options.stop);

    Json literal_spans = Json::array();
    std::size_t previous_end = 0;
    for (const auto span : rendered.literal_spans) {
        if (span.begin >= span.end || span.begin < previous_end || span.end > rendered.text.size()) {
            throw std::logic_error("production literal span is invalid");
        }
        literal_spans.push_back({{"byte_offset", span.begin}, {"byte_length", span.end - span.begin}});
        previous_end = span.end;
    }
    Json token_pieces = Json::array();
    for (const auto id : actual.token_ids) {
        if (id < 0 || static_cast<std::uint32_t>(id) >= resources.public_token_count) {
            throw std::logic_error("prepared token escaped the public token domain");
        }
        token_pieces.push_back(token_piece(*resources.tokenizer, id));
    }
    Json special_tokens = Json::array();
    for (std::uint32_t id = 0; id < resources.public_token_count; ++id) {
        if (resources.tokenizer->is_special_token(static_cast<int>(id))) {
            special_tokens.push_back(token_piece(*resources.tokenizer, static_cast<int>(id)));
        }
    }
    Json thinking = nullptr;
    if (input.options.enable_thinking) { thinking = *input.options.enable_thinking; }
    Json output{
        {"schema_version", 1},
        {"host_only", true},
        {"rendered_utf8", rendered.text},
        {"rendered_byte_count", rendered.text.size()},
        {"literal_spans", std::move(literal_spans)},
        {"token_ids", actual.token_ids},
        {"token_count", actual.token_ids.size()},
        {"token_pieces", std::move(token_pieces)},
        {"public_token_count", resources.public_token_count},
        {"physical_vocab_size", physical_count},
        {"excluded_physical_token_range", {resources.public_token_count, physical_count}},
        {"special_tokens", std::move(special_tokens)},
        {"default_stop_token_ids", frontend.default_stop_policy().token_ids},
        {"caller_stop_token_ids", options.stop.token_ids},
        {"effective_stop_token_ids", effective.token_ids},
        {"starts_in_reasoning", actual.starts_in_reasoning},
        {"prompt_options", {{"enable_thinking", thinking}, {"continuation", "new_assistant_turn"}}},
        {"stop_policy", {{"include_model_defaults", options.stop.include_model_defaults},
                          {"publish_stop_token", effective.publish_stop_token},
                          {"strings", Json::array()}}},
        {"validation", {{"rendered_replay_matches_prepared_ids", true},
                         {"public_domain_check", "production parse_resources"},
                         {"scope", "text_only_without_tools_or_explicit_cache_markers"}}},
        {"input", {{"kind", !options.messages.empty() ? "messages_json" : "user_text"},
                    {"message_count", input.messages.size()},
                    {"resources_directory", path_utf8(options.resources)},
                    {"messages_path", path_utf8(options.messages)},
                    {"prompt_file", path_utf8(options.prompt_file)},
                    {"max_context", options.max_context}}},
    };
    return output;
}

} // namespace frontend_export_tool

int main(int argc, char** argv) {
    try {
        const auto options = frontend_export_tool::parse_options(argc, argv);
        // Strict JSON serialization preserves rendered UTF-8 and escapes NUL,
        // CR/LF and controls. Per-token invalid UTF-8 was retained as hex above.
        const std::string json = frontend_export_tool::export_prompt(options).dump(2) + "\n";
        if (options.output.empty()) {
            std::cout << json;
            if (!std::cout) { throw std::runtime_error("could not write JSON to stdout"); }
        } else {
            std::ofstream stream(options.output, std::ios::binary | std::ios::trunc);
            if (!stream || !stream.write(json.data(), static_cast<std::streamsize>(json.size()))) {
                throw std::runtime_error("could not write JSON output");
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "frontend-export: " << error.what() << '\n';
        return 1;
    }
}
