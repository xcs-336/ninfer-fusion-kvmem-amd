#pragma once

#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::product {

inline std::string parse_device_profile_mode(std::string_view value) {
    if (value == "auto" || value == "off" || value == "calibrate") {
        return std::string(value);
    }
    throw std::invalid_argument("--device-profile must be auto, off or calibrate");
}

} // namespace ninfer::product
