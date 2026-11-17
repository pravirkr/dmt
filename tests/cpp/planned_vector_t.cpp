#include "dmt/planned_vector.hpp"

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>

// Deferred allocation (planned_vector.hpp): a sequence of calls made while
// deferring, then materialised, gives the vector the same calls make at
// once; a sequence a plan cannot represent throws instead.

namespace dmt {

namespace {

using planned::DeferScope;
using planned::HostPlan;
using planned::PendingList;
using Vec = planned::PlannedHostVector<float>;

/// Runs the calls at once (eager), or deferred: `in_scope` inside the
/// deferral scope, `after` after it but before materialising.
std::vector<float> eager(const std::function<void(Vec&)>& calls) {
    Vec v;
    calls(v);
    return {v.begin(), v.end()};
}

std::vector<float> deferred(const std::function<void(Vec&)>& in_scope,
                            const std::function<void(Vec&)>& after = {}) {
    PendingList list;
    Vec v;
    {
        const DeferScope scope(&list);
        in_scope(v);
        CHECK(v.pending());
    }
    if (after) {
        after(v);
        CHECK(v.pending()); // still deferring into the same list
    }
    const auto planned = v.size();
    list.materialize();
    CHECK_FALSE(v.pending());
    CHECK(v.size() == planned);
    return {v.begin(), v.end()};
}

} // namespace

TEST_CASE("Planned vectors replay deferred calls exactly", "[planned]") {
    SECTION("assign, then a larger resize with the same value") {
        const auto f = [](Vec& v) {
            v.assign(4, 7.0F);
            v.resize(6, 7.0F);
        };
        CHECK(deferred(f) == eager(f));
    }
    SECTION("resize, then a larger resize with the default value") {
        const auto f = [](Vec& v) {
            v.resize(3);
            v.resize(5, 0.0F);
        };
        CHECK(deferred(f) == eager(f));
    }
    SECTION("assign, then resize_zeroed: all zero") {
        const auto f = [](Vec& v) {
            v.assign(4, 7.0F);
            v.resize_zeroed(6);
        };
        CHECK(deferred(f) == eager(f));
        CHECK(deferred(f) == std::vector<float>(6, 0.0F));
    }
    SECTION("assign, then a host copy") {
        const std::vector<float> host{1.0F, 2.0F, 3.0F};
        const auto f = [&](Vec& v) {
            v.assign(8, 7.0F);
            v = host;
        };
        CHECK(deferred(f) == eager(f));
        CHECK(deferred(f) == host);
    }
    SECTION("a host copy, then a larger resize") {
        const std::vector<float> host{1.0F, 2.0F};
        const auto f = [&](Vec& v) {
            v = host;
            v.resize(4, 9.0F);
        };
        CHECK(deferred(f) == eager(f));
    }
    SECTION("calls after the scope keep deferring") {
        const std::vector<float> host{4.0F, 5.0F};
        const auto first = [](Vec& v) { v.resize(3, 2.0F); };
        CHECK(deferred(first, [](Vec& v) { v.assign(5, 6.0F); }) ==
              eager([](Vec& v) {
                  v.resize(3, 2.0F);
                  v.assign(5, 6.0F);
              }));
        CHECK(deferred(first, [&](Vec& v) { v = host; }) == host);
        CHECK(deferred(first, [](Vec& v) { v.resize_zeroed(4); }) ==
              std::vector<float>(4, 0.0F));
    }
    SECTION("a larger resize with a different value throws") {
        PendingList list;
        Vec v;
        const DeferScope scope(&list);
        v.resize(3, 2.0F);
        CHECK_THROWS_AS(v.resize(5), std::logic_error);
        Vec w;
        w.resize_zeroed(3);
        CHECK_THROWS_AS(w.resize(5, 1.0F), std::logic_error);
        Vec u;
        u.assign(2, 1.0F);
        CHECK_THROWS_AS(u.resize(4, 3.0F), std::logic_error);
    }
}

TEST_CASE("HostPlan sizes recorded buffers", "[planned]") {
    HostPlan plan;
    plan.defer(true);
    std::vector<float> a;
    std::vector<int> b;
    plan.assign(a, 4, 1.0F);
    plan.resize(b, 3, 2);
    plan.assign(a, 6, 5.0F); // a buffer sized again replaces its plan
    CHECK(a.empty());
    CHECK(plan.size(a) == 6);
    CHECK(plan.size(b) == 3);
    plan.materialize();
    CHECK(a == std::vector<float>(6, 5.0F));
    CHECK(b == std::vector<int>(3, 2));
    CHECK(plan.size(a) == 6);
    plan.resize(b, 4, 9); // later calls forward
    CHECK(b == std::vector<int>{2, 2, 2, 9});
}

} // namespace dmt
