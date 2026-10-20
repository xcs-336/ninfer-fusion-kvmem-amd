#include "amd_hardware_class.h"

#include <array>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>

int main() {
    using ninfer::amd::context_cost_hardware_class;
    int failures = 0;
    const auto expect = [&](bool valid, std::string_view label) {
        if (!valid) { std::cerr << label << '\n'; ++failures; }
    };
    const std::string expected = "amd-radeon-rx-9070-gfx1201";
    for (const auto name : {"AMD Radeon RX 9070", "Radeon RX 9070", "  AMD  Radeon RX 9070  ",
                            "AMD AMD Radeon RX 9070"}) {
        expect(context_cost_hardware_class(name, "gfx1201") == expected,
               "RX9070 native class lost architecture or duplicated vendor");
    }
    const auto positive = context_cost_hardware_class("AMD Radeon RX 9070", "gfx1201:sramecc+:xnack-");
    const auto negative = context_cost_hardware_class("AMD Radeon RX 9070", "gfx1201:sramecc-:xnack-");
    expect(positive == expected + ":sramecc+:xnack-", "gfx feature bytes/order were changed");
    expect(positive != negative, "different gfx feature flags collided");
    expect(context_cost_hardware_class("Instinct MI210", "gfx90a:xnack+") ==
               "amd-instinct-mi210-gfx90a:xnack+", "hex gfx processor was rejected or changed");
    expect(positive.find("nvidia") == std::string::npos && positive.find("-sm") == std::string::npos,
           "native AMD class contains NVIDIA capability identity");
    const std::array<std::pair<std::string_view, std::string_view>, 12> invalid{{
        {"", "gfx1201"}, {" -- ", "gfx1201"}, {"AMD", ""}, {"AMD", "gfx"},
        {"AMD", "sm120"}, {"AMD", "gfx1201/xnack-"}, {"AMD", "gfx1201:xnack"},
        {"AMD", "gfx1201:+"}, {"AMD", "gfx1201:xnack-:"}, {"AMD", "gfx1201:xnack--"},
        {std::string_view("AMD\0Radeon", 10), "gfx1201"},
        {"AMD", std::string_view("gfx1201\0", 8)},
    }};
    for (const auto& [name, architecture] : invalid) {
        bool rejected = false;
        try { (void)context_cost_hardware_class(name, architecture); }
        catch (const std::invalid_argument&) { rejected = true; }
        expect(rejected, "invalid native hardware identity was accepted");
    }
    std::cout << (failures == 0 ? "PASS" : "FAIL") << " native AMD hardware identity\n";
    return failures == 0 ? 0 : 1;
}
