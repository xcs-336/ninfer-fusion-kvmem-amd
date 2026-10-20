"""Expose the existing real Engine profile mode after the native metadata view."""


def once(source, before, after):
    if source.count(before) != 1:
        raise ValueError('Expected one device-profile source anchor: ' + before[:100])
    return source.replace(before, after)


def cli_header(source):
    return once(source, '    int device                   = 0;',
        '    int device                   = 0;\n    std::string device_profile{"auto"};')


def cli_parser(source):
    source = '#include "product/device_profile_options.h"\n' + source
    source = once(source,
        '           "  --device N                    CUDA device index (default 0)\\n"',
        '           "  --device N                    CUDA device index (default 0)\\n"\n'
        '           "  --device-profile MODE         auto (default), off or calibrate; controls\\n"\n'
        '           "                                actual route-profile lookup/calibration\\n"')
    return once(source, '        } else if (arg == "--devices") {',
        '''        } else if (arg == "--device-profile") {
            options.device_profile = product::parse_device_profile_mode(value(arg));
        } else if (arg == "--devices") {''')


def cli_main(source):
    source = once(source, '        engine_options.device                   = cli.device;',
        '        engine_options.device                   = cli.device;\n'
        '        engine_options.device_profile           = cli.device_profile;')
    return once(source, '    logger->info("build {}", NINFER_BUILD_ID);',
        '    logger->info("build {}", NINFER_BUILD_ID);\n'
        '    logger->info("device route profile mode {}", cli.device_profile);')


def bench_header(source):
    source = once(source, '    int device            = 0;',
        '    int device            = 0;\n    std::string device_profile{"auto"};')
    return once(source, '    int device_id = 0;',
        '    int device_id = 0;\n    std::string device_profile{"auto"};')


def bench_main(source):
    source = once(source, '        engine_options.device        = options.device;',
        '        engine_options.device        = options.device;\n'
        '        engine_options.device_profile = options.device_profile;')
    return once(source, '        env.max_context              = max_context;',
        '        env.max_context              = max_context;\n'
        '        env.device_profile           = options.device_profile;')


def bench_support(source):
    source = '#include "product/device_profile_options.h"\n' + source
    source = once(source,
        '        << "  --device <id>               CUDA device ordinal (default: 0)\\n"',
        '        << "  --device <id>               CUDA device ordinal (default: 0)\\n"\n'
        '        << "  --device-profile <auto|off|calibrate>  actual route profile mode (default: auto)\\n"')
    source = once(source, '        } else if (arg == "--no-cuda-graph") {',
        '''        } else if (arg == "--device-profile") {
            options.device_profile = product::parse_device_profile_mode(value("--device-profile"));
        } else if (arg == "--no-cuda-graph") {''')
    source = once(source, '        << "  artifact:   " << env.artifact_path',
        '        << "  route profile: " << env.device_profile << \'\\n\'\n'
        '        << "  artifact:   " << env.artifact_path')
    source = once(source, r'''        << "    \"max_context\": " << env.max_context << ",\n"''',
        r'''        << "    \"device_profile\": \"" << json_escape(env.device_profile) << "\",\n"
        << "    \"max_context\": " << env.max_context << ",\n"''')
    source = once(source, '           "context,rope_yarn_factor,prefill_chunk,"',
        '           "context,rope_yarn_factor,prefill_chunk,device_profile,"')
    return once(source, '            << env.rope_yarn_factor << \',\' << env.prefill_chunk',
        '            << env.rope_yarn_factor << \',\' << env.prefill_chunk << \',\' << env.device_profile')


def perplexity(source):
    source = '#include "product/device_profile_options.h"\n' + source
    source = once(source, '    int device                = 0;',
        '    int device                = 0;\n    std::string device_profile{"auto"};')
    source = once(source, '           "       [--context N] [--stride N | --disjoint] [--device N]\\n"',
        '           "       [--context N] [--stride N | --disjoint] [--device N]\\n"\n'
        '           "       [--device-profile auto|off|calibrate] (default auto)\\n"')
    source = once(source, '        } else if (option == "--fast-prefill-kernel") {',
        '''        } else if (option == "--device-profile") {
            out.device_profile = ninfer::product::parse_device_profile_mode(value("--device-profile"));
        } else if (option == "--fast-prefill-kernel") {''')
    source = once(source, '    engine_options.device           = options.device;',
        '    engine_options.device           = options.device;\n'
        '    engine_options.device_profile   = options.device_profile;')
    return once(source, '          {"device", options.device},',
        '          {"device", options.device},\n          {"device_profile", options.device_profile},')


def cli_tests(source):
    tests = '''    failures += check(parse({"ninfer-cli", "model.ninfer", "--prompt", "x"}).device_profile == "auto",
                      "default profile mode changed");
    for (const auto* mode : {"auto", "off", "calibrate"}) {
        failures += check(parse({"ninfer-cli", "model.ninfer", "--prompt", "x", "--device-profile", mode})
                              .device_profile == mode, "profile mode was not preserved");
    }
    failures += check(rejects([] { (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "x",
                                                "--device-profile", "pretend"}); }),
                      "invalid profile mode was accepted");
    failures += check(rejects([] { (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "x",
                                                "--device-profile"}); }),
                      "missing profile mode was accepted");
    failures += check(ninfer::cli::usage_text("ninfer-cli").find("--device-profile") != std::string::npos,
                      "profile mode is absent from CLI help");
'''
    return once(source, '    int failures = 0;\n', '    int failures = 0;\n' + tests)


def bench_tests(source):
    tests = '''    failures += expect(parse_for_test({"ninfer_bench", "--weights", "model.ninfer"}).device_profile == "auto",
                       "default route profile mode");
    for (const auto* mode : {"auto", "off", "calibrate"}) {
        failures += expect(parse_for_test({"ninfer_bench", "--weights", "model.ninfer", "--device-profile", mode})
                               .device_profile == mode, "explicit route profile mode");
    }
    failures += expect_throws<std::invalid_argument>([] {
        (void)parse_for_test({"ninfer_bench", "--weights", "model.ninfer", "--device-profile", "pretend"});
    }, "invalid route profile mode");
    failures += expect_throws<std::invalid_argument>([] {
        (void)parse_for_test({"ninfer_bench", "--weights", "model.ninfer", "--device-profile"});
    }, "missing route profile mode");
    failures += expect(qb::usage_text("ninfer_bench").find("--device-profile") != std::string::npos,
                       "profile mode in benchmark help");
'''
    source = once(source, 'int test_cli_contract() {\n    int failures                  = 0;',
        'int test_cli_contract() {\n    int failures                  = 0;\n' + tests)
    source = once(source, '    env.device_id                = 0;',
        '    env.device_id                = 0;\n    env.device_profile           = "off";')
    source = once(source, '    failures += expect(report.at("config").at("rope_yarn_factor") == 2.5,',
        '    failures += expect(report.at("config").at("device_profile") == "off", "report route profile mode");\n'
        '    failures += expect(report.at("config").at("rope_yarn_factor") == 2.5,')
    source = once(source, '    failures += expect(table.find("Qwen3_5ForCausalLM") != std::string::npos, "table target");',
        '    failures += expect(table.find("route profile: off") != std::string::npos, "human report profile mode");\n'
        '    failures += expect(table.find("Qwen3_5ForCausalLM") != std::string::npos, "table target");')
    return once(source, '{"model_name", "artifact_path", "proposal_head",',
        '{"device_profile", "model_name", "artifact_path", "proposal_head",')


TRANSFORMS = (
    ('apps/cli/options.h', cli_header), ('apps/cli/options.cpp', cli_parser),
    ('apps/cli/main.cpp', cli_main),
    ('bench/inference/ninfer_bench_support.h', bench_header),
    ('bench/inference/ninfer_bench_support.cpp', bench_support),
    ('bench/inference/ninfer_bench.cpp', bench_main),
    ('apps/perplexity/main.cpp', perplexity),
    ('tests/test_cli_options.cpp', cli_tests),
    ('tests/test_ninfer_bench_support.cpp', bench_tests),
)
