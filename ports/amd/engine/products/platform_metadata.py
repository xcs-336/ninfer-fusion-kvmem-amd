"""True HIP product metadata; the original pure CUDA formatter fixtures remain."""


def once(source, before, after):
    if source.count(before) != 1:
        raise ValueError('Expected one product metadata anchor: ' + before[:100])
    return source.replace(before, after)


FIELDS = '''    // Platform metadata is emitted for the actual backend only. The CUDA
    // fields below also serve the original independent CUDA formatter fixtures.
    std::string backend{"cuda"};
    std::string gcn_arch_name;
    std::string hip_compile_version;
    std::string hip_runtime_version;
    std::string hip_driver_version;
'''


def environment_header(source):
    return once(source, '    std::string gpu_name;', '    std::string gpu_name;\n' + FIELDS)


def benchmark_main(source):
    start = source.index('std::string cuda_version_string(int version) {')
    end = source.index('void require_cuda(', start)
    source = source[:start] + '''void fill_hip_environment(ninfer::bench::BenchEnvironment& env, int device) {
    env.device_id = device;
    env.backend = "hip";
    env.hip_compile_version = ninfer::amd::hip_compile_version_string();
    int runtime_version = 0;
    if (hipRuntimeGetVersion(&runtime_version) == hipSuccess) {
        env.hip_runtime_version = ninfer::amd::hip_version_string(runtime_version);
    }
    int driver_version = 0;
    if (hipDriverGetVersion(&driver_version) == hipSuccess) {
        env.hip_driver_version = ninfer::amd::hip_version_string(driver_version);
    }
    hipDeviceProp_t properties{};
    if (hipGetDeviceProperties(&properties, device) == hipSuccess) {
        env.gpu_name = properties.name;
        env.gcn_arch_name = properties.gcnArchName;
    }
}

''' + source[end:]
    source = once(source, 'fill_cuda_environment(env, options.device);',
                  'fill_hip_environment(env, options.device);')
    return '#include "product/amd_environment.h"\n' + source


def benchmark_support(source):
    source = once(source, '''        << "  cuda:       runtime " << env.cuda_runtime_version << ", driver "
        << env.cuda_driver_version << '\\n'
''', '''        << (env.backend == "hip"
                ? "  hip:        runtime " + env.hip_runtime_version + ", driver " + env.hip_driver_version +
                    " (" + env.gcn_arch_name + ")\\n"
                : "  cuda:       runtime " + env.cuda_runtime_version + ", driver " + env.cuda_driver_version + "\\n")
''')
    function = source.index('std::string format_json(')
    anchor = source.index('    std::ostringstream out;', function)
    hardware = '''    std::ostringstream hardware;
    hardware << "{\\"gpu_name\\": \\"" << json_escape(env.gpu_name);
    if (env.backend == "hip") {
        hardware << "\\", \\"backend\\": \\"hip\\", \\"gcn_arch_name\\": \\"" << json_escape(env.gcn_arch_name)
                 << "\\", \\"hip_compile_version\\": \\"" << json_escape(env.hip_compile_version)
                 << "\\", \\"hip_runtime_version\\": \\"" << json_escape(env.hip_runtime_version)
                 << "\\", \\"hip_driver_version\\": \\"" << json_escape(env.hip_driver_version);
    } else {
        hardware << "\\", \\"cuda_runtime_version\\": \\"" << json_escape(env.cuda_runtime_version)
                 << "\\", \\"cuda_driver_version\\": \\"" << json_escape(env.cuda_driver_version);
    }
    hardware << "\\", \\"device_id\\": " << env.device_id << "}";

'''
    source = source[:anchor] + hardware + source[anchor:]
    before = r'''        << "  \"environment\": {\"gpu_name\": \"" << json_escape(env.gpu_name)
        << "\", \"cuda_runtime_version\": \"" << json_escape(env.cuda_runtime_version)
        << "\", \"cuda_driver_version\": \"" << json_escape(env.cuda_driver_version)
        << "\", \"device_id\": " << env.device_id << "},\n"
'''
    source = once(source, before,
                  r'''        << "  \"environment\": " << hardware.str() << ",\n"
''')
    return source


def request_log(source):
    start = source.index('    record["environment"] =')
    end = source.index('    record["argv"]', start)
    original = source[start:end]
    source = source[:start] + '''    if (environment.backend == "hip") {
        record["environment"] = Json{{"device", environment.device},
            {"gpu_name", environment.gpu_name}, {"gpu_uuid", environment.gpu_uuid},
            {"total_device_memory_bytes", environment.total_device_memory_bytes},
            {"backend", "hip"}, {"gcn_arch_name", environment.gcn_arch_name},
            {"hip_compile_version", environment.hip_compile_version},
            {"hip_runtime_version", environment.hip_runtime_version},
            {"hip_driver_version", environment.hip_driver_version}};
    } else {
''' + original + '    }\n' + source[end:]
    start = source.index('ServerLogEnvironment query_server_log_environment(int device) {')
    end = source.index('JsonlRequestLog::JsonlRequestLog(', start)
    source = source[:start] + '''ServerLogEnvironment query_server_log_environment(int device) {
    ServerLogEnvironment environment;
    environment.device = device;
    environment.backend = "hip";
    environment.hip_compile_version = ninfer::amd::hip_compile_version_string();
    int runtime_version = 0;
    if (hipRuntimeGetVersion(&runtime_version) == hipSuccess) {
        environment.hip_runtime_version = ninfer::amd::hip_version_string(runtime_version);
    }
    int driver_version = 0;
    if (hipDriverGetVersion(&driver_version) == hipSuccess) {
        environment.hip_driver_version = ninfer::amd::hip_version_string(driver_version);
    }
    hipDeviceProp_t properties{};
    if (hipGetDeviceProperties(&properties, device) == hipSuccess) {
        environment.gpu_name = properties.name;
        environment.gpu_uuid = cuda_uuid_string(properties.uuid);
        environment.total_device_memory_bytes = properties.totalGlobalMem;
        environment.gcn_arch_name = properties.gcnArchName;
    }
    return environment;
}

''' + source[end:]
    return '#include "product/amd_environment.h"\n' + source


def startup_text(source):
    for before, after, expected in [('initializing CUDA', 'initializing HIP', 1),
                                    ('CUDA initialized', 'HIP initialized', 1),
                                    ('preparing CUDA graphs', 'preparing HIP graphs', 1),
                                    ('CUDA graphs ready', 'HIP graphs ready', 2)]:
        if source.count(before) != expected:
            raise ValueError('Unexpected original startup text: ' + before)
        source = source.replace(before, after)
    return source


def calibrate_hardware_class(source):
    source = once(source, '''ninfer::runtime::context_cost_hardware_class(
            properties.name, properties.major, properties.minor)''',
        '''ninfer::amd::context_cost_hardware_class(properties.name, properties.gcnArchName)''')
    source = once(source, '(%d SMs)', '(%d compute units)')
    return '#include "runtime/amd_hardware_class.h"\n' + source
