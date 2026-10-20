#include "evictable_weight_pool.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

using Clock = std::chrono::steady_clock;

void hip_check(hipError_t result, const char* expression) {
    if (result != hipSuccess) {
        throw std::runtime_error(std::string(expression) + " failed: " + hipGetErrorString(result));
    }
}

#define NINFER_WEIGHT_HIP_CHECK(expression) ::ninfer::hip_check((expression), #expression)

std::size_t chunk_round_up(std::size_t value) {
    constexpr std::size_t chunk = EvictableWeightPool::kChunkBytes;
    if (value > std::numeric_limits<std::size_t>::max() - (chunk - 1)) {
        throw std::invalid_argument("evictable weight extent overflows chunk rounding");
    }
    return (value + chunk - 1) / chunk * chunk;
}

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

} // namespace

struct EvictableWeightPool::Impl {
    int device;
    hipStream_t compute_stream;
    hipStream_t transfer_stream;
    Config config{};
    std::size_t arena_reserved = 0;
    std::size_t tail_begin = 0;
    std::size_t window_capacity = 0;
    std::size_t mirror_begin = 0;
    std::size_t mirror_extent = 0;
    std::size_t lent_extent = 0;
    std::byte* home = nullptr;
    void* mirror = nullptr;
    bool mirror_captured = false;
    bool poisoned = false;

    explicit Impl(const DeviceContext& context)
        : device(context.device), compute_stream(context.stream),
          transfer_stream(context.transfer_stream) {}

    ~Impl() noexcept {
        // Constructor failures also destroy Impl, so partially acquired resources are released.
        // Cleanup cannot propagate a failed device/context status through another exception.
        int previous = -1;
        const bool have_previous = hipGetDevice(&previous) == hipSuccess;
        if (!have_previous || previous != device) { (void)hipSetDevice(device); }
        if (mirror != nullptr) { (void)hipHostFree(mirror); }
        if (home != nullptr) { (void)hipFree(home); }
        if (have_previous && previous != device) { (void)hipSetDevice(previous); }
    }
};

bool EvictableWeightPool::supported(const DeviceContext& device) {
    hipDeviceProp_t properties{};
    return hipGetDeviceProperties(&properties, device.device) == hipSuccess &&
           properties.totalGlobalMem != 0;
}

EvictableWeightPool::EvictableWeightPool(DeviceContext& device, const Config& config)
    : impl_(std::make_unique<Impl>(device)) {
    if (config.arena_bytes == 0) {
        throw std::invalid_argument("evictable pool arena must not be empty");
    }
    if (config.evictable_tail_bytes == 0 || config.evictable_tail_bytes > config.arena_bytes) {
        throw std::invalid_argument("evictable pool tail must be a nonempty arena suffix");
    }
    Impl& impl = *impl_;
    impl.config = config;
    impl.arena_reserved = chunk_round_up(config.arena_bytes);
    // Match the original physical suffix exactly; a chunk overlapping a resident object cannot
    // be borrowed. In particular, unaligned arena/tail sizes do not make the boundary chunk free.
    impl.tail_begin = chunk_round_up(config.arena_bytes - config.evictable_tail_bytes);
    if (impl.tail_begin >= impl.arena_reserved) {
        throw std::invalid_argument("evictable pool tail covers no whole chunk");
    }
    const DeviceBinding binding(impl.device);
    NINFER_WEIGHT_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&impl.home), impl.arena_reserved));
}

EvictableWeightPool::~EvictableWeightPool() = default;

DeviceSpan EvictableWeightPool::arena() const noexcept {
    return {impl_->home, impl_->config.arena_bytes};
}

std::size_t EvictableWeightPool::evictable_tail_bytes() const noexcept {
    return impl_->arena_reserved - impl_->tail_begin;
}

std::size_t EvictableWeightPool::window_capacity_bytes() const noexcept {
    return impl_->window_capacity;
}

std::size_t EvictableWeightPool::mirror_bytes() const noexcept { return impl_->mirror_extent; }
bool EvictableWeightPool::mirror_captured() const noexcept { return impl_->mirror_captured; }
bool EvictableWeightPool::transaction_open() const noexcept { return impl_->lent_extent != 0; }
bool EvictableWeightPool::poisoned() const noexcept { return impl_->poisoned; }

void EvictableWeightPool::capture_window_mirror(std::size_t window_capacity_bytes,
                                               hipStream_t stream) {
    Impl& impl = *impl_;
    if (impl.mirror_captured) {
        throw std::logic_error("evictable pool window mirror was already captured");
    }
    if (window_capacity_bytes == 0) {
        throw std::invalid_argument("evictable pool window capacity must be positive");
    }
    const std::size_t window = chunk_round_up(window_capacity_bytes);
    if (window > impl.arena_reserved - impl.tail_begin) {
        throw std::invalid_argument("evictable pool window exceeds the evictable tail");
    }
    const std::size_t begin = impl.arena_reserved - window;
    const std::size_t extent = impl.config.arena_bytes > begin ? impl.config.arena_bytes - begin : 0;
    const DeviceBinding binding(impl.device);
    void* mirror = nullptr;
    if (extent != 0) {
        NINFER_WEIGHT_HIP_CHECK(hipHostMalloc(&mirror, extent, hipHostMallocDefault));
        try {
            NINFER_WEIGHT_HIP_CHECK(
                hipMemcpyAsync(mirror, impl.home + begin, extent, hipMemcpyDeviceToHost, stream));
            NINFER_WEIGHT_HIP_CHECK(hipStreamSynchronize(stream));
        } catch (...) {
            // A submitted async copy cannot outlive its destination on the exception path.
            (void)hipStreamSynchronize(stream);
            (void)hipHostFree(mirror);
            throw;
        }
    }
    // Commit metadata only after the complete mirror exists; a refused/failed capture stays closed.
    impl.mirror = mirror;
    impl.window_capacity = window;
    impl.mirror_begin = begin;
    impl.mirror_extent = extent;
    impl.mirror_captured = true;
}

EvictableWeightPool::Transaction EvictableWeightPool::evict(std::size_t bytes,
                                                           hipStream_t stream) {
    Impl& impl = *impl_;
    if (!impl.mirror_captured) {
        throw std::logic_error("evictable pool has no window mirror; capture it after load");
    }
    if (impl.poisoned) {
        throw std::runtime_error("evictable pool is poisoned by a failed restore");
    }
    if (impl.lent_extent != 0) {
        throw std::logic_error("evictable pool transaction is already open");
    }
    if (bytes == 0) { throw std::invalid_argument("evictable pool cannot evict zero bytes"); }
    const std::size_t extent = chunk_round_up(bytes);
    if (extent > impl.window_capacity) {
        throw std::invalid_argument("evictable pool request exceeds the window capacity: " +
                                    std::to_string(bytes) + " bytes");
    }
    const DeviceBinding binding(impl.device);
    NINFER_WEIGHT_HIP_CHECK(hipStreamSynchronize(impl.compute_stream));
    NINFER_WEIGHT_HIP_CHECK(hipStreamSynchronize(impl.transfer_stream));
    const auto start = Clock::now();
    impl.lent_extent = extent;
    return Transaction(*this, DeviceSpan{impl.home + impl.arena_reserved - extent, extent}, stream,
                       seconds_since(start));
}

void EvictableWeightPool::restore(Transaction& transaction) noexcept {
    Impl& impl = *impl_;
    if (impl.lent_extent == 0) { return; }
    const auto start = Clock::now();
    try {
        const DeviceBinding binding(impl.device);
        const std::size_t begin = impl.arena_reserved - impl.lent_extent;
        const std::size_t end = impl.config.arena_bytes;
        if (end > begin) {
            NINFER_WEIGHT_HIP_CHECK(hipMemcpyAsync(
                impl.home + begin,
                static_cast<const std::byte*>(impl.mirror) + (begin - impl.mirror_begin),
                end - begin, hipMemcpyHostToDevice, transaction.stream_));
        }
        // Also drains loan work when the loan consists entirely of allocation padding.
        NINFER_WEIGHT_HIP_CHECK(hipStreamSynchronize(transaction.stream_));
    } catch (const std::exception& error) {
        impl.poisoned = true;
        std::fprintf(stderr, "ninfer: native HIP weight restore failed (%s); the weights are no "
                             "longer trustworthy\n", error.what());
    } catch (...) {
        impl.poisoned = true;
        std::fprintf(stderr, "ninfer: native HIP weight restore failed; the weights are no "
                             "longer trustworthy\n");
    }
    impl.lent_extent = 0;
    transaction.stats_.restore_seconds = seconds_since(start);
}

EvictableWeightPool::Transaction::Transaction(EvictableWeightPool& pool, DeviceSpan leased,
                                              hipStream_t stream, double evict_seconds) noexcept
    : pool_(&pool), leased_(leased), stream_(stream) {
    stats_.evict_seconds = evict_seconds;
    stats_.mapped_bytes = leased.bytes;
}

EvictableWeightPool::Transaction::~Transaction() { close(); }

EvictableWeightPool::Transaction::Transaction(Transaction&& other) noexcept
    : pool_(other.pool_), leased_(other.leased_), stream_(other.stream_), stats_(other.stats_) {
    other.pool_ = nullptr;
}

EvictableWeightPool::Transaction&
EvictableWeightPool::Transaction::operator=(Transaction&& other) noexcept {
    if (this != &other) {
        close();
        pool_ = other.pool_;
        leased_ = other.leased_;
        stream_ = other.stream_;
        stats_ = other.stats_;
        other.pool_ = nullptr;
    }
    return *this;
}

void EvictableWeightPool::Transaction::close() noexcept {
    if (pool_ == nullptr) { return; }
    pool_->restore(*this);
    pool_ = nullptr;
}

} // namespace ninfer
