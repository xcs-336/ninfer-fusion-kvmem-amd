#pragma once
// Candidate header: finite concrete model boundary, not a backend/plugin protocol.
#include "models/qwen3_5/program/program.h"
#include <type_traits>
#include <utility>
namespace ninfer::models::qwen3_8_flash_next { class Program; }
namespace ninfer::runtime::segmented_worker_detail {
class Lifetime {
    bool tracked_;
public:
    explicit Lifetime(bool tracked) noexcept:tracked_(tracked) {
        if(tracked_)segmented_worker_counters::starts.fetch_add(1);
    }
    ~Lifetime(){if(tracked_)segmented_worker_counters::exits.fetch_add(1);}
};
template<class Program> class Controller {
    static_assert(std::is_same_v<Program,models::qwen3_8_flash_next::Program>,
        "Only the concrete ordinary FlashNext Program bypasses the Qwen controller bridge");
public:
    explicit Controller(Program&) noexcept {}
    Controller(Controller&&) noexcept=default;
    void consume() noexcept {}
    void retire() noexcept {}
    void before_ready() noexcept {}
    [[nodiscard]] bool tracked() const noexcept{return false;}
};
template<> class Controller<models::qwen3_5::Program> {
    using Program=models::qwen3_5::Program;
    Program& program_;
    Program::StartupControllerCapability capability_;
    bool accepted_=false;
    bool tracked_=false;
public:
    explicit Controller(Program& program):program_(program),capability_(program.release_startup_controller()),
        tracked_(static_cast<bool>(capability_)){}
    Controller(const Controller&)=delete;
    Controller(Controller&& other) noexcept:program_(other.program_),
        capability_(std::move(other.capability_)),accepted_(std::exchange(other.accepted_,false)),
        tracked_(std::exchange(other.tracked_,false)){}
    void consume() {program_.accept_startup_controller(std::move(capability_));accepted_=true;}
    void before_ready(){program_.worker_startup_ready_test();}
    [[nodiscard]] bool tracked() const noexcept{return tracked_;}
    void retire() {
        if(!accepted_)return;
        // Mark before the checked operation: an error is retained, never retried/rebound.
        accepted_=false;program_.retire_worker_controller();
    }
};
} // namespace ninfer::runtime::segmented_worker_detail
