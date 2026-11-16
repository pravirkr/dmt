#pragma once

// Deferred device allocation for memory estimates (B24).
//
// An engine built with FDMTEngineConfig::defer_allocation runs its whole
// constructor (every decision, every host table) inside a DeferScope. A
// PlannedVector<T> (a thrust::device_vector<T>) then records what the
// constructor asks of it (a size and fill value, or a host upload) instead
// of allocating, and reports that size from size(), so the engine's
// get_memory_usage() is unchanged code. Each pending vector registers in the
// scope's list; the engine's allocate() materialises the list. Outside a
// DeferScope a PlannedVector is a plain thrust::device_vector.

#include <algorithm>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <thrust/device_vector.h>

namespace dmt::gpu_utils {

class PlannedBase;

/// The vectors a deferred construction left pending, in registration order.
/// Owned by the engine; entries are updated when a pending vector moves and
/// cleared when it is destroyed.
class PendingList {
public:
    PendingList()                              = default;
    PendingList(const PendingList&)            = delete;
    PendingList& operator=(const PendingList&) = delete;
    PendingList(PendingList&&)                 = delete;
    PendingList& operator=(PendingList&&)      = delete;
    ~PendingList();

    void add(PlannedBase* v) { m_items.push_back(v); }
    void replace(PlannedBase* from, PlannedBase* to) noexcept {
        std::replace(m_items.begin(), m_items.end(), from, to);
    }
    /// Allocates (and fills or uploads) every pending vector.
    void materialize();
    [[nodiscard]] bool empty() const noexcept {
        return std::all_of(m_items.begin(), m_items.end(),
                           [](const PlannedBase* p) { return p == nullptr; });
    }

private:
    std::vector<PlannedBase*> m_items;
};

/// The active deferral scope of this thread (nullptr: allocate as usual).
inline PendingList*& active_pending() noexcept {
    thread_local PendingList* active = nullptr;
    return active;
}

/// RAII: vectors touched while alive defer into `list` (nullptr: no-op).
class DeferScope {
public:
    explicit DeferScope(PendingList* list) noexcept : m_prev(active_pending()) {
        active_pending() = list;
    }
    ~DeferScope() { active_pending() = m_prev; }
    DeferScope(const DeferScope&)            = delete;
    DeferScope& operator=(const DeferScope&) = delete;
    DeferScope(DeferScope&&)                 = delete;
    DeferScope& operator=(DeferScope&&)      = delete;

private:
    PendingList* m_prev;
};

class PlannedBase {
public:
    PlannedBase()                              = default;
    PlannedBase(const PlannedBase&)            = delete;
    PlannedBase& operator=(const PlannedBase&) = delete;
    PlannedBase(PlannedBase&&)                 = delete;
    PlannedBase& operator=(PlannedBase&&)      = delete;
    virtual ~PlannedBase()                     = default;
    virtual void materialize()                 = 0;
    virtual void detach() noexcept             = 0;
};

inline PendingList::~PendingList() {
    for (auto* p : m_items) {
        if (p != nullptr) {
            p->detach();
        }
    }
}

inline void PendingList::materialize() {
    for (auto*& p : m_items) {
        if (p != nullptr) {
            PlannedBase* v = p;
            p              = nullptr;
            v->materialize();
        }
    }
    m_items.clear();
}

template <typename T>
class PlannedVector : public thrust::device_vector<T>, public PlannedBase {
    using Base = thrust::device_vector<T>;

public:
    PlannedVector() = default;
    explicit PlannedVector(std::size_t n) { resize(n); }
    PlannedVector(std::size_t n, const T& v) { resize(n, v); }
    PlannedVector(const PlannedVector& other)
        : Base(static_cast<const Base&>(other)),
          PlannedBase() {
        if (other.m_list != nullptr) {
            throw std::logic_error("PlannedVector: copy of a pending vector");
        }
    }
    PlannedVector(PlannedVector&& other) noexcept
        : Base(std::move(static_cast<Base&>(other))),
          PlannedBase() {
        take(other);
    }
    PlannedVector& operator=(const PlannedVector& other) {
        if (this != &other) {
            if (other.m_list != nullptr) {
                throw std::logic_error(
                    "PlannedVector: copy of a pending vector");
            }
            drop();
            Base::operator=(static_cast<const Base&>(other));
        }
        return *this;
    }
    PlannedVector& operator=(PlannedVector&& other) noexcept {
        if (this != &other) {
            drop();
            Base::operator=(std::move(static_cast<Base&>(other)));
            take(other);
        }
        return *this;
    }
    ~PlannedVector() override { drop(); }

    template <typename A>
    PlannedVector& operator=(const std::vector<T, A>& host) {
        if (auto* list = active_pending(); list != nullptr) {
            defer(list);
            m_host.assign(host.begin(), host.end());
            m_fill.reset();
            m_size = host.size();
        } else {
            drop();
            Base::operator=(host);
        }
        return *this;
    }

    void resize(std::size_t n) { resize_impl(n, std::nullopt); }
    void resize(std::size_t n, const T& v) { resize_impl(n, v); }
    void assign(std::size_t n, const T& v) {
        if (auto* list = active_pending(); list != nullptr) {
            defer(list);
            m_host.clear();
            m_fill   = v;
            m_size   = n;
            m_assign = true;
        } else {
            drop();
            Base::assign(n, v);
        }
    }
    void clear() {
        drop();
        Base::clear();
    }

    /// The planned size while pending, else the allocated one.
    [[nodiscard]] std::size_t size() const noexcept {
        return m_list != nullptr ? m_size : Base::size();
    }
    [[nodiscard]] bool empty() const noexcept { return size() == 0; }
    [[nodiscard]] bool pending() const noexcept { return m_list != nullptr; }

    void materialize() override {
        m_list = nullptr;
        if (!m_host.empty()) {
            Base::operator=(m_host);
        } else if (m_assign) {
            Base::assign(m_size, m_fill.value_or(T{}));
        } else if (m_fill) {
            Base::resize(m_size, *m_fill);
        } else {
            Base::resize(m_size);
        }
        reset_plan();
    }
    void detach() noexcept override {
        m_list = nullptr;
        reset_plan();
    }

private:
    void resize_impl(std::size_t n, std::optional<T> v) {
        auto* list = active_pending();
        if (list == nullptr && m_list == nullptr) {
            if (v) {
                Base::resize(n, *v);
            } else {
                Base::resize(n);
            }
            return;
        }
        if (list == nullptr) {
            // Resized after the deferred construction, before allocate():
            // keep deferring into the same list.
            list = m_list;
        }
        defer(list);
        if (!m_host.empty()) {
            m_host.resize(n, v.value_or(T{}));
        }
        if (v && !m_assign && n > m_size) {
            // thrust::resize fills only the new tail; a deferred buffer is
            // empty, so the whole of it is the new tail.
            m_fill = v;
        }
        m_size = n;
    }
    void defer(PendingList* list) {
        if (m_list == nullptr) {
            if (Base::size() != 0) {
                // Keep what was allocated before the scope (not expected).
                m_host.resize(Base::size());
                thrust::copy(Base::begin(), Base::end(), m_host.begin());
                m_size = Base::size();
                Base::clear();
                Base::shrink_to_fit();
            }
            m_list = list;
            list->add(this);
        } else if (m_list != list) {
            throw std::logic_error("PlannedVector: two deferral scopes");
        }
    }
    void drop() noexcept {
        if (m_list != nullptr) {
            m_list->replace(this, nullptr);
            m_list = nullptr;
        }
        reset_plan();
    }
    void take(PlannedVector& other) noexcept {
        m_list   = other.m_list;
        m_size   = other.m_size;
        m_fill   = other.m_fill;
        m_assign = other.m_assign;
        m_host   = std::move(other.m_host);
        if (m_list != nullptr) {
            m_list->replace(&other, this);
        }
        other.m_list = nullptr;
        other.reset_plan();
    }
    void reset_plan() noexcept {
        m_size   = 0;
        m_assign = false;
        m_fill.reset();
        m_host.clear();
    }

    PendingList* m_list{nullptr};
    std::size_t m_size{0};
    std::optional<T> m_fill;
    bool m_assign{false};
    std::vector<T> m_host;
};

} // namespace dmt::gpu_utils
