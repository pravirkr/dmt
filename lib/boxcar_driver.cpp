#include <cstdint>
#include <memory>
#include <span>
#include <utility>

#include "dmt/boxcar.hpp"

namespace dmt::algorithms::detail {
namespace {

// FDMT: every selected level's rows, the root merged inside the search.
class FDMTDriver final : public BoxcarDriver {
public:
    explicit FDMTDriver(std::unique_ptr<FDMTEngine> fdmt)
        : m_fdmt(std::move(fdmt)) {}
    void search(std::span<const float> in, FDMTSearchSink& sink) override {
        m_fdmt->execute_search(in, sink);
    }
    void search(DeviceSpan<const float> in,
                FDMTSearchSink& sink,
                Stream stream) override {
        m_fdmt->execute_search(in, sink, stream);
    }
    void search(std::span<const uint8_t> in,
                SizeType nbits,
                FDMTSearchSink& sink) override {
        m_fdmt->execute_search(in, nbits, sink);
    }
    void search(DeviceSpan<const uint8_t> in,
                SizeType nbits,
                FDMTSearchSink& sink,
                Stream stream) override {
        m_fdmt->execute_search(in, nbits, sink, stream);
    }
    void reset(Stream stream) override { m_fdmt->reset_search_history(stream); }
    [[nodiscard]] FDMTMemoryUsage memory() const noexcept override {
        return m_fdmt->get_memory_usage();
    }
    [[nodiscard]] SizeType chunk_nsamps() const noexcept override {
        return m_fdmt->get_chunk_nsamps();
    }
    [[nodiscard]] SizeType fuse_levels() const noexcept override {
        return m_fdmt->get_fuse_levels();
    }
    [[nodiscard]] std::string search_path() const override {
        return m_fdmt->search_path();
    }

private:
    std::unique_ptr<FDMTEngine> m_fdmt;
};

// FDMT-FFT: the root rows of each overlap-save segment, float input only.
class FDMTFFTDriver final : public BoxcarDriver {
public:
    explicit FDMTFFTDriver(std::unique_ptr<FDMTFFTEngine> fft)
        : m_fft(std::move(fft)) {}
    void search(std::span<const float> in, FDMTSearchSink& sink) override {
        m_fft->execute_search(in, sink);
    }
    void search(DeviceSpan<const float> in,
                FDMTSearchSink& sink,
                Stream stream) override {
        m_fft->execute_search(in, sink, stream);
    }
    void reset(Stream stream) override { m_fft->reset_search_history(stream); }
    [[nodiscard]] FDMTMemoryUsage memory() const noexcept override {
        return m_fft->memory_usage();
    }
    [[nodiscard]] SizeType chunk_nsamps() const noexcept override {
        return m_fft->segment_nsamps();
    }
    [[nodiscard]] SizeType fuse_levels() const noexcept override { return 0; }

private:
    std::unique_ptr<FDMTFFTEngine> m_fft;
};

} // namespace

std::unique_ptr<BoxcarDriver>
make_boxcar_driver(std::unique_ptr<FDMTEngine> fdmt) {
    return std::make_unique<FDMTDriver>(std::move(fdmt));
}

std::unique_ptr<BoxcarDriver>
make_boxcar_driver(std::unique_ptr<FDMTFFTEngine> fft) {
    return std::make_unique<FDMTFFTDriver>(std::move(fft));
}

} // namespace dmt::algorithms::detail
