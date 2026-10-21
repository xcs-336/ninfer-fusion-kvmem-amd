    // Candidate-only, one-shot opaque startup controller. No Core/HIP types leak.
    class StartupControllerCapability {
    public:
        StartupControllerCapability() noexcept;
        ~StartupControllerCapability();
        StartupControllerCapability(StartupControllerCapability&&) noexcept;
        StartupControllerCapability& operator=(StartupControllerCapability&&) noexcept;
        StartupControllerCapability(const StartupControllerCapability&)=delete;
        StartupControllerCapability& operator=(const StartupControllerCapability&)=delete;
        [[nodiscard]] explicit operator bool() const noexcept;
    private:
        struct State;
        std::unique_ptr<State> state_;
        friend class Program;
    };
    using SegmentedStorageAccounting=::ninfer::SegmentedEngineStorageSnapshot;
    [[nodiscard]] StartupControllerCapability release_startup_controller();
    void accept_startup_controller(StartupControllerCapability&&);
    void retire_worker_controller();
    void set_segmented_startup_test(SegmentedWorkerStartupTest);
    void worker_startup_ready_test();
    [[nodiscard]] SegmentedStorageAccounting segmented_storage_accounting() const;
