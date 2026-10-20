// Bounded first-execution probes through the unchanged public Linear dispatch. These use the
// upstream packed fixtures, independent FP64 oracle, numerical criteria, guards and preservation
// checks. They complement the full original suites with smaller first-execution fixtures.
#include "ops/linear/linear_test_common.h"

#include <array>
#include <exception>
#include <iostream>
#include <string_view>

namespace {
using namespace ninfer;
using namespace ninfer::test::linear;

constexpr std::array kA16{
    Invocation{1}, Invocation{3}, Invocation{17, CallForm::Policy, ops::LinearPolicy::A16Only, true},
    Invocation{33},
};
// The registered Q6 vision contraction admits four-token groups, including graph replay.
constexpr std::array kQ6A16{
    Invocation{4}, Invocation{8}, Invocation{16}, Invocation{32},
    Invocation{36, CallForm::Policy, ops::LinearPolicy::A16Only, true},
};
constexpr std::array kA8{
    Invocation{1, CallForm::Policy, ops::LinearPolicy::AllowA8},
    Invocation{17, CallForm::Policy, ops::LinearPolicy::AllowA8},
    Invocation{33, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
};
constexpr std::array kA4{
    Invocation{1, CallForm::Policy, ops::LinearPolicy::AllowA4},
    Invocation{17, CallForm::Policy, ops::LinearPolicy::AllowA4, true},
    Invocation{33, CallForm::Policy, ops::LinearPolicy::AllowA4},
};
constexpr std::array kInteger{
    Invocation{1, CallForm::Policy, ops::LinearPolicy::AllowA8Int},
    Invocation{17, CallForm::Policy, ops::LinearPolicy::AllowA8Int, true},
    Invocation{33, CallForm::Policy, ops::LinearPolicy::AllowA8Int},
};

struct Family {
    std::string_view name;
    WeightGenerator generator;
    std::int32_t n, k;
    std::uint32_t seed;
};
constexpr std::array kFamilies{
    Family{"q4", make_q4_g64_fp16_weight, 1024, 5120, 1701},
    Family{"q5", make_q5_g64_fp16_weight, 1024, 5120, 1703},
    Family{"q6", make_q6_g64_fp16_weight, 1152, 1536, 1705},
    Family{"q8", make_q8_g32_fp16_weight, 2048, 4096, 1707},
    Family{"t2", make_t2_g128_fp16_weight, 1024, 5120, 1709},
    Family{"fp8", make_fp8_weight, 5120, 6144, 1711},
    Family{"nvfp4", make_nvfp4_weight, 5120, 6144, 1713},
};
} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: linear_public_probe q4|q5|q6|q8|t2|fp8|nvfp4\n";
        return 2;
    }
    if (!ninfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable HIP device\n";
        return 77;
    }
    try {
        for (const Family& family : kFamilies) {
            if (family.name != argv[1]) { continue; }
            const auto run = [&](std::string_view label, ActivationCompute compute,
                                 std::span<const Invocation> invocations) {
                return run_shape(label, compute, family.generator,
                                 {family.n, family.k, family.seed, Comparison::SampledRows,
                                  true, invocations});
            };
            const std::span<const Invocation> a16_invocations =
                family.name == "q6" ? std::span<const Invocation>(kQ6A16)
                                    : std::span<const Invocation>(kA16);
            int failures = run("PUBLIC_A16", ActivationCompute::A16, a16_invocations);
            if (family.name == "fp8") {
                failures += run("PUBLIC_A8", ActivationCompute::A8, kA8);
            } else if (family.name == "nvfp4") {
                failures += run("PUBLIC_A4", ActivationCompute::A4, kA4);
            } else if (family.name == "t2") {
                failures += run("PUBLIC_INTEGER_A8", ActivationCompute::A8, kInteger);
            }
            std::cout << (failures == 0 ? "PASS" : "FAIL") << " public Linear " << family.name
                      << " bounded probe (full original family suite remains required)\n";
            return failures == 0 ? 0 : 1;
        }
        std::cerr << "unknown family: " << argv[1] << '\n';
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "public Linear probe: " << error.what() << '\n';
        return 1;
    }
}
