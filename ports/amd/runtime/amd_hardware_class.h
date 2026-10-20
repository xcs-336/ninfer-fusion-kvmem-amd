#pragma once

#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::amd {

// Profile keys describe the actual native architecture. HIP's numeric major/minor
// fields are not NVIDIA SM capabilities, and target-ID feature flags distinguish
// devices that must not accidentally share a measured profile.
[[nodiscard]] inline std::string context_cost_hardware_class(std::string_view gpu_name,
                                                            std::string_view gcn_arch_name) {
    const auto ascii_alnum = [](unsigned char value) {
        return (value >= 'a' && value <= 'z') || (value >= 'A' && value <= 'Z') ||
               (value >= '0' && value <= '9');
    };
    if (gpu_name.empty() || gpu_name.find('\0') != std::string_view::npos) {
        throw std::invalid_argument("AMD hardware class requires a nonempty GPU name without NUL");
    }
    if (!gcn_arch_name.starts_with("gfx") || gcn_arch_name.find('\0') != std::string_view::npos) {
        throw std::invalid_argument("AMD hardware class requires a native gfx architecture");
    }
    std::size_t position = 3;
    const std::size_t base_begin = position;
    while (position < gcn_arch_name.size()) {
        const char value = gcn_arch_name[position];
        if (!((value >= '0' && value <= '9') || (value >= 'a' && value <= 'f'))) { break; }
        ++position;
    }
    if (position == base_begin) {
        throw std::invalid_argument("AMD hardware class requires a nonempty gfx processor");
    }
    while (position < gcn_arch_name.size()) {
        if (gcn_arch_name[position++] != ':') {
            throw std::invalid_argument("AMD hardware class has an invalid gfx feature separator");
        }
        const std::size_t feature_begin = position;
        while (position < gcn_arch_name.size() &&
               (ascii_alnum(static_cast<unsigned char>(gcn_arch_name[position])) ||
                gcn_arch_name[position] == '_')) {
            ++position;
        }
        if (position == feature_begin || position == gcn_arch_name.size() ||
            (gcn_arch_name[position] != '+' && gcn_arch_name[position] != '-')) {
            throw std::invalid_argument("AMD hardware class requires signed gfx feature flags");
        }
        ++position;
    }

    std::string slug;
    slug.reserve(gpu_name.size());
    for (const unsigned char value : gpu_name) {
        if (ascii_alnum(value)) {
            slug.push_back(static_cast<char>(value >= 'A' && value <= 'Z' ? value + ('a' - 'A') : value));
        } else if (!slug.empty() && slug.back() != '-') {
            slug.push_back('-');
        }
    }
    while (!slug.empty() && slug.back() == '-') { slug.pop_back(); }
    if (slug.empty()) {
        throw std::invalid_argument("AMD hardware class requires an alphanumeric GPU name");
    }
    while (slug.starts_with("amd-amd-") || slug == "amd-amd") { slug.erase(0, 4); }
    if (slug != "amd" && !slug.starts_with("amd-")) { slug.insert(0, "amd-"); }
    return slug + '-' + std::string(gcn_arch_name);
}

} // namespace ninfer::amd
