#pragma once

// Internal tests: wraps the transform of every FDMTBoxcar built while a
// BoxcarDriverHook is alive (detail::g_boxcar_driver_wrap). An armed hook
// makes the next search throw, after the engine committed the call's block
// noise; with `lags`, the CPU skewed path's row lags are recorded per level.

#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "dmt/boxcar.hpp"
#include "dmt/fdmt_search.hpp"

namespace dmt::test {

struct BoxcarDriverHook {
    bool armed = false;
    /// [level][row] lags of the host float searches (empty: unlagged).
    std::vector<std::vector<uint64_t>>* lags = nullptr;
    /// Rows per level (the lagged rows' count), with `lags`.
    std::vector<SizeType> level_rows;

    BoxcarDriverHook() {
        if (current() != nullptr) {
            throw std::logic_error("BoxcarDriverHook: one at a time");
        }
        current()                                = this;
        algorithms::detail::g_boxcar_driver_wrap = &wrap;
    }
    ~BoxcarDriverHook() {
        algorithms::detail::g_boxcar_driver_wrap = nullptr;
        current()                                = nullptr;
    }
    BoxcarDriverHook(const BoxcarDriverHook&)            = delete;
    BoxcarDriverHook& operator=(const BoxcarDriverHook&) = delete;
    BoxcarDriverHook(BoxcarDriverHook&&)                 = delete;
    BoxcarDriverHook& operator=(BoxcarDriverHook&&)      = delete;

private:
    using BoxcarDriver   = algorithms::detail::BoxcarDriver;
    using FDMTSearchSink = algorithms::detail::FDMTSearchSink;

    static BoxcarDriverHook*& current() {
        static BoxcarDriverHook* hook = nullptr;
        return hook;
    }

    void trip() {
        if (armed) {
            armed = false;
            throw std::runtime_error("injected search failure");
        }
    }

    class LagRecorder final : public FDMTSearchSink {
    public:
        LagRecorder(FDMTSearchSink& inner, BoxcarDriverHook& hook)
            : m_inner(inner),
              m_hook(hook) {}
        void consume(SizeType level,
                     const algorithms::detail::FDMTSearchInput& in) override {
            auto& lags = *m_hook.lags;
            if (in.row_lag != nullptr && level < m_hook.level_rows.size()) {
                if (lags.size() <= level) {
                    lags.resize(level + 1);
                }
                lags[level].assign(in.row_lag,
                                   in.row_lag + m_hook.level_rows[level]);
            }
            m_inner.consume(level, in);
        }

    private:
        FDMTSearchSink& m_inner;
        BoxcarDriverHook& m_hook;
    };

    class Driver final : public BoxcarDriver {
    public:
        Driver(std::unique_ptr<BoxcarDriver> inner, BoxcarDriverHook& hook)
            : m_inner(std::move(inner)),
              m_hook(hook) {}
        void search(std::span<const float> in, FDMTSearchSink& sink) override {
            m_hook.trip();
            if (m_hook.lags != nullptr) {
                LagRecorder rec(sink, m_hook);
                m_inner->search(in, rec);
                return;
            }
            m_inner->search(in, sink);
        }
        void search(DeviceSpan<const float> in,
                    FDMTSearchSink& sink,
                    Stream stream) override {
            m_hook.trip();
            m_inner->search(in, sink, stream);
        }
        void search(std::span<const uint8_t> in,
                    SizeType nbits,
                    FDMTSearchSink& sink) override {
            m_hook.trip();
            m_inner->search(in, nbits, sink);
        }
        void search(DeviceSpan<const uint8_t> in,
                    SizeType nbits,
                    FDMTSearchSink& sink,
                    Stream stream) override {
            m_hook.trip();
            m_inner->search(in, nbits, sink, stream);
        }
        void search_zero(FDMTSearchSink& sink) override {
            m_hook.trip();
            m_inner->search_zero(sink);
        }
        void reset(Stream stream) override { m_inner->reset(stream); }
        void allocate() override { m_inner->allocate(); }
        [[nodiscard]] bool stages_host_input() const noexcept override {
            return m_inner->stages_host_input();
        }
        bool set_search_keep(const uint8_t* keep, float fill) override {
            return m_inner->set_search_keep(keep, fill);
        }
        [[nodiscard]] bool search_zero_supported() const noexcept override {
            return m_inner->search_zero_supported();
        }
        [[nodiscard]] algorithms::FDMTMemoryUsage
        memory() const noexcept override {
            return m_inner->memory();
        }
        [[nodiscard]] SizeType chunk_nsamps() const noexcept override {
            return m_inner->chunk_nsamps();
        }
        [[nodiscard]] SizeType fuse_levels() const noexcept override {
            return m_inner->fuse_levels();
        }
        [[nodiscard]] algorithms::detail::FDMTPath
        path() const noexcept override {
            return m_inner->path();
        }
        [[nodiscard]] std::string search_path() const override {
            return m_inner->search_path();
        }
        [[nodiscard]] uint64_t search_latency() const noexcept override {
            return m_inner->search_latency();
        }

    private:
        std::unique_ptr<BoxcarDriver> m_inner;
        BoxcarDriverHook& m_hook;
    };

    static std::unique_ptr<BoxcarDriver>
    wrap(std::unique_ptr<BoxcarDriver> inner) {
        return std::make_unique<Driver>(std::move(inner), *current());
    }
};

} // namespace dmt::test
