#pragma once

// Deferred allocation for memory estimates.
//
// An engine built with FDMTEngineConfig::defer_allocation runs its whole
// constructor (every decision, every table) inside a DeferScope. A
// Planned<V> (a V: std::vector or thrust::device_vector) then records what
// the constructor asks of it (a size and fill value, or a host copy)
// instead of allocating, and reports that size from size(), so the engine's
// memory accounting is unchanged code. Each pending vector registers in the
// scope's list; the engine's allocate() materialises the list. Outside a
// DeferScope a Planned<V> is a plain V.

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstring>
#include <functional>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace dmt::planned {

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
    /// Allocates (and fills or copies) every pending vector.
    void materialize();
    [[nodiscard]] bool empty() const noexcept {
        return std::all_of(m_items.begin(), m_items.end(),
                           [](const PlannedBase* p) { return p == nullptr; });
    }
    /// Vectors that deferred into the list since it was last materialised
    /// (destroyed ones included).
    [[nodiscard]] std::size_t registered() const noexcept {
        return m_items.size();
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

/// Sets every byte of `v` to zero (specialised for device vectors).
template <typename V> struct ZeroBytes {
    static void apply(V& v) {
        if (!v.empty()) {
            std::memset(static_cast<void*>(v.data()), 0,
                        v.size() * sizeof(typename V::value_type));
        }
    }
};

template <typename V> class Planned : public V, public PlannedBase {
    using Base = V;
    using T    = typename V::value_type;

public:
    Planned() = default;
    explicit Planned(std::size_t n) { resize(n); }
    Planned(std::size_t n, const T& v) { resize(n, v); }
    Planned(const Planned& other)
        : Base(static_cast<const Base&>(other)),
          PlannedBase() {
        if (other.m_list != nullptr) {
            throw std::logic_error("Planned: copy of a pending vector");
        }
    }
    Planned(Planned&& other) noexcept
        : Base(std::move(static_cast<Base&>(other))),
          PlannedBase() {
        take(other);
    }
    Planned& operator=(const Planned& other) {
        if (this != &other) {
            if (other.m_list != nullptr) {
                throw std::logic_error("Planned: copy of a pending vector");
            }
            drop();
            Base::operator=(static_cast<const Base&>(other));
        }
        return *this;
    }
    Planned& operator=(Planned&& other) noexcept {
        if (this != &other) {
            drop();
            Base::operator=(std::move(static_cast<Base&>(other)));
            take(other);
        }
        return *this;
    }
    ~Planned() override { drop(); }

    template <typename A> Planned& operator=(const std::vector<T, A>& host) {
        if (auto* list = active_pending(); list != nullptr || pending()) {
            defer(list != nullptr ? list : m_list);
            m_host.assign(host.begin(), host.end());
            m_fill.reset();
            m_assign = false;
            m_zero   = false;
            m_size   = host.size();
        } else {
            drop();
            Base::assign(host.begin(), host.end());
        }
        return *this;
    }

    void resize(std::size_t n) { resize_impl(n, std::nullopt); }
    void resize(std::size_t n, const T& v) { resize_impl(n, v); }
    /// resize(n), then every byte zero (padding included).
    void resize_zeroed(std::size_t n) {
        if (auto* list = active_pending(); list != nullptr || pending()) {
            defer(list != nullptr ? list : m_list);
            m_host.clear();
            m_fill.reset();
            m_assign = false;
            m_size   = n;
            m_zero   = true;
        } else {
            Base::resize(n);
            ZeroBytes<Base>::apply(*this);
        }
    }
    void assign(std::size_t n, const T& v) {
        if (auto* list = active_pending(); list != nullptr || pending()) {
            defer(list != nullptr ? list : m_list);
            m_host.clear();
            m_fill   = v;
            m_size   = n;
            m_assign = true;
            m_zero   = false;
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
            Base::assign(m_host.begin(), m_host.end());
        } else if (m_assign) {
            Base::assign(m_size, m_fill.value_or(T{}));
        } else if (m_fill) {
            Base::resize(m_size, *m_fill);
        } else {
            Base::resize(m_size);
            if (m_zero) {
                ZeroBytes<Base>::apply(*this);
            }
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
        } else if (n > m_size) {
            // resize() fills only the new tail. A plan holds one fill for
            // the whole buffer: the tail's, if the buffer was empty, else
            // it must match the fill already planned.
            const T tail = v.value_or(T{});
            if (m_size == 0) {
                m_fill   = v;
                m_assign = false;
                m_zero   = false;
            } else if (!same_value(tail, planned_fill())) {
                throw std::logic_error(
                    "Planned: deferred resize with a different fill");
            }
        }
        m_size = n;
    }
    // The value every planned element gets (no host copy).
    [[nodiscard]] T planned_fill() const { return m_fill.value_or(T{}); }
    static bool same_value(const T& a, const T& b) {
        if constexpr (std::equality_comparable<T>) {
            return a == b;
        } else {
            return std::memcmp(&a, &b, sizeof(T)) == 0;
        }
    }
    void defer(PendingList* list) {
        if (m_list == nullptr) {
            if (Base::size() != 0) {
                // Keep what was allocated before the scope (not expected).
                m_host = std::vector<T>(Base::begin(), Base::end());
                m_size = Base::size();
                Base::clear();
                Base::shrink_to_fit();
            }
            m_list = list;
            list->add(this);
        } else if (m_list != list) {
            throw std::logic_error("Planned: two deferral scopes");
        }
    }
    void drop() noexcept {
        if (m_list != nullptr) {
            m_list->replace(this, nullptr);
            m_list = nullptr;
        }
        reset_plan();
    }
    void take(Planned& other) noexcept {
        m_list   = other.m_list;
        m_size   = other.m_size;
        m_fill   = other.m_fill;
        m_assign = other.m_assign;
        m_zero   = other.m_zero;
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
        m_zero   = false;
        m_fill.reset();
        m_host.clear();
    }

    PendingList* m_list{nullptr};
    std::size_t m_size{0};
    std::optional<T> m_fill;
    bool m_assign{false};
    bool m_zero{false};
    std::vector<T> m_host;
};

/// A std::vector<T> whose allocation a DeferScope can defer.
template <typename T> using PlannedHostVector = Planned<std::vector<T>>;

/**
 * Deferred host buffers kept as plain std::vector members (no wrapper type
 * on the hot path): while deferring, resize() and assign() record the
 * call instead of allocating, size() reports the planned size, and
 * materialize() replays the calls in order. Not deferring, they forward.
 * Holds references to the vectors: owned by the engine that owns them.
 */
class HostPlan {
public:
    void defer(bool on) noexcept { m_defer = on; }
    [[nodiscard]] bool deferring() const noexcept { return m_defer; }

    template <typename T, typename A>
    void resize(std::vector<T, A>& v,
                std::size_t n,
                const std::type_identity_t<T>& fill = T{}) {
        if (!m_defer) {
            v.resize(n, fill);
            return;
        }
        record(&v, n, [&v, n, fill] { v.resize(n, fill); });
    }
    template <typename T, typename A>
    void assign(std::vector<T, A>& v,
                std::size_t n,
                const std::type_identity_t<T>& fill) {
        if (!m_defer) {
            v.assign(n, fill);
            return;
        }
        record(&v, n, [&v, n, fill] { v.assign(n, fill); });
    }
    /// The planned size of `v` while pending, else its size.
    template <typename T, typename A>
    [[nodiscard]] std::size_t size(const std::vector<T, A>& v) const noexcept {
        for (const auto& item : m_items) {
            if (item.vec == &v) {
                return item.size;
            }
        }
        return v.size();
    }
    /// Allocates every recorded buffer; later calls forward.
    void materialize() {
        m_defer = false;
        for (auto& item : m_items) {
            item.alloc();
        }
        m_items.clear();
    }

private:
    struct Item {
        const void* vec{nullptr};
        std::size_t size{0};
        std::function<void()> alloc;
    };
    void record(const void* vec, std::size_t n, std::function<void()> alloc) {
        // A buffer sized again replaces its plan (deferred buffers are empty,
        // so the last call alone decides).
        for (auto& item : m_items) {
            if (item.vec == vec) {
                item.size  = n;
                item.alloc = std::move(alloc);
                return;
            }
        }
        m_items.push_back({vec, n, std::move(alloc)});
    }

    bool m_defer = false;
    std::vector<Item> m_items;
};

} // namespace dmt::planned
