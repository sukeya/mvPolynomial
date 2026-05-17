#include "mvPolynomial/mvPolynomial.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

using MP2 = mvPolynomial::MVPolynomial<int, double, 2>;
using MP3 = mvPolynomial::MVPolynomial<int, double, 3>;

template <typename Poly>
typename Poly::mapped_type NaivePointEval(const Poly& p, const typename Poly::coord_type& x) {
  auto sum = typename Poly::mapped_type{0};
  for (const auto& [index, coeff] : p) {
    auto term = coeff;
    for (int axis = 0; axis < Poly::dim; ++axis) {
      term *= std::pow(x[axis], index[axis]);
    }
    sum += term;
  }
  return sum;
}

template <typename Poly>
Poly NaiveCompose(const Poly& p, const Poly& mvp, int axis) {
  auto composed = Poly(p.get_allocator());
  for (const auto& [index, coeff] : p) {
    auto index_without_axis  = index;
    index_without_axis[axis] = 0;
    composed += Poly(
                    {
                        {index_without_axis, coeff}
    },
                    p.get_allocator()
                ) *
                mvp.pow(index[axis]);
  }
  return composed;
}

template <typename Poly, typename URNG>
Poly MakeRandomPolynomial(URNG& rng, int term_count, int max_degree, int max_abs_coeff) {
  auto poly = Poly();

  auto degree_dist = std::uniform_int_distribution<int>(0, max_degree);
  auto coeff_dist  = std::uniform_int_distribution<int>(-max_abs_coeff, max_abs_coeff);
  for (int term = 0; term < term_count; ++term) {
    typename Poly::index_type index;
    index.setZero();
    for (int axis = 0; axis < Poly::dim; ++axis) {
      index(axis) = degree_dist(rng);
    }

    auto coeff = coeff_dist(rng);
    if (coeff == 0) {
      continue;
    }

    if (poly.contains(index)) {
      poly.set(index, poly.get(index) + coeff);
    } else {
      poly.set(index, coeff);
    }
  }

  if (poly == Poly()) {
    typename Poly::index_type index;
    index.setZero();
    index(0) = 1;
    poly.set(index, 1);
  }

  return poly;
}

template <typename Poly, typename URNG>
typename Poly::coord_type MakeRandomPoint(URNG& rng, int min_coord, int max_coord) {
  typename Poly::coord_type point;
  point.setZero();
  auto coord_dist = std::uniform_int_distribution<int>(min_coord, max_coord);
  for (int axis = 0; axis < Poly::dim; ++axis) {
    point(axis) = static_cast<typename Poly::mapped_type>(coord_dist(rng));
  }
  return point;
}

template <typename Poly>
std::string DescribePolynomial(const Poly& p) {
  auto oss   = std::ostringstream{};
  auto first = true;
  oss << "{";
  for (const auto& [index, coeff] : p) {
    if (!first) {
      oss << ", ";
    }
    first = false;
    oss << "[";
    for (int axis = 0; axis < Poly::dim; ++axis) {
      if (axis > 0) {
        oss << ",";
      }
      oss << index[axis];
    }
    oss << "]=" << coeff;
  }
  oss << "}";
  return oss.str();
}

template <typename Poly>
std::string DescribePoint(const typename Poly::coord_type& point) {
  auto oss = std::ostringstream{};
  oss << "[";
  for (int axis = 0; axis < Poly::dim; ++axis) {
    if (axis > 0) {
      oss << ",";
    }
    oss << point[axis];
  }
  oss << "]";
  return oss.str();
}

template <typename T>
class CountingAllocator {
 public:
  using value_type = T;

  CountingAllocator() : allocations_(std::make_shared<std::size_t>(0)) {}

  explicit CountingAllocator(std::shared_ptr<std::size_t> allocations) : allocations_(std::move(allocations)) {}

  template <typename U>
  CountingAllocator(const CountingAllocator<U>& other) : allocations_(other.allocations_) {}

  [[nodiscard]] T* allocate(std::size_t n) {
    *allocations_ += n;
    return std::allocator<T>{}.allocate(n);
  }

  void deallocate(T* p, std::size_t n) noexcept { std::allocator<T>{}.deallocate(p, n); }

  template <typename U>
  bool operator==(const CountingAllocator<U>& other) const noexcept {
    return allocations_ == other.allocations_;
  }

  template <typename U>
  bool operator!=(const CountingAllocator<U>& other) const noexcept {
    return !(*this == other);
  }

  std::shared_ptr<std::size_t> allocations_;
};

using CountingPairAllocator = CountingAllocator<std::pair<const Eigen::Array2i, double>>;
using CountingMP2           = mvPolynomial::MVPolynomial<int, double, 2, CountingPairAllocator>;

TEST_CASE("constructor", "[mvPolynomial]") {
  SECTION("default") {
    auto ans = std::vector<std::pair<Eigen::Array2i, double>>({
        {{0, 0}, 0},
    });
    auto m   = MP2();

    REQUIRE(m.size() == ans.size());
    for (size_t i = 0; i < ans.size(); ++i) {
      REQUIRE(m.get(ans[i].first) == ans[i].second);
    }
  }

  SECTION("range") {
    auto ans = std::vector<std::pair<Eigen::Array2i, double>>({
        {{0, 0}, 1},
        {{1, 0}, 2},
        {{0, 1}, 3},
        {{1, 1}, 4},
        {{2, 0}, 5},
        {{0, 2}, 6},
    });

    auto m = MP2(ans.begin(), ans.end());

    REQUIRE(m.size() == ans.size());
    for (size_t i = 0; i < ans.size(); ++i) {
      REQUIRE(m.get(ans[i].first) == ans[i].second);
    }
  }

  SECTION("empty range is canonical zero polynomial") {
    auto empty = std::vector<std::pair<Eigen::Array2i, double>>{};
    auto m     = MP2(empty.begin(), empty.end());

    REQUIRE(m.size() == 1);
    REQUIRE(m == MP2());
    REQUIRE(m.contains(Eigen::Array2i::Zero()));
    REQUIRE(m.get(Eigen::Array2i::Zero()) == 0);
    REQUIRE(m(Eigen::Vector2d::Zero()) == 0);
  }

  SECTION("initializer_list") {
    auto ans = std::vector<std::pair<Eigen::Array2i, double>>({
        {{0, 0}, 1},
        {{1, 0}, 2},
        {{0, 1}, 3},
        {{1, 1}, 4},
        {{2, 0}, 5},
        {{0, 2}, 6},
    });

    auto m = MP2({
        {{0, 0}, 1},
        {{1, 0}, 2},
        {{0, 1}, 3},
        {{1, 1}, 4},
        {{2, 0}, 5},
        {{0, 2}, 6},
    });

    REQUIRE(m.size() == ans.size());
    for (size_t i = 0; i < ans.size(); ++i) {
      REQUIRE(m.get(ans[i].first) == ans[i].second);
    }
  }

  SECTION("empty initializer_list is canonical zero polynomial") {
    auto m = MP2({});

    REQUIRE(m.size() == 1);
    REQUIRE(m == MP2());
    REQUIRE(m.contains(Eigen::Array2i::Zero()));
    REQUIRE(m.get(Eigen::Array2i::Zero()) == 0);
    REQUIRE(m(Eigen::Vector2d({2, 3})) == 0);
  }

  SECTION("zero coefficient terms are normalized away at construction") {
    auto m = MP2({
        {{1, 0}, 0},
        {{0, 1}, 0},
        {{0, 0}, 0},
    });

    REQUIRE(m.size() == 1);
    REQUIRE(m == MP2());
    REQUIRE(m.contains(Eigen::Array2i::Zero()));
    REQUIRE(m.get(Eigen::Array2i::Zero()) == 0);
    REQUIRE(m(Eigen::Vector2d({2, 3})) == 0);
  }

  SECTION("negative index is rejected") {
    REQUIRE_THROWS_AS(
        MP2({
            {{-1, 0}, 1},
    }),
        std::invalid_argument
    );
  }
}

TEST_CASE("invariants", "[mvPolynomial]") {
  auto require_canonical_zero = [](const MP2& p) {
    REQUIRE(p.size() == 1);
    REQUIRE(p == MP2());
    REQUIRE(p.contains(Eigen::Array2i::Zero()));
    REQUIRE(p.get(Eigen::Array2i::Zero()) == 0);
  };

  SECTION("subtracting self stays canonical zero polynomial") {
    auto p = MP2({
        {{0, 0}, 1},
        {{1, 0}, 2},
        {{0, 1}, 3},
    });

    p -= p;

    require_canonical_zero(p);
  }

  SECTION("multiplying by zero stays canonical zero polynomial") {
    auto p = MP2({
        {{0, 0}, 1},
        {{1, 0}, 2},
        {{0, 1}, 3},
    });

    p *= 0.0;

    require_canonical_zero(p);
  }

  SECTION("set rejects negative index for lvalue and rvalue keys") {
    auto p       = MP2();
    auto bad_key = Eigen::Array2i({-1, 0});

    REQUIRE_THROWS_AS(p.set(bad_key, 1.0), std::invalid_argument);
    REQUIRE_THROWS_AS(p.set(Eigen::Array2i({0, -1}), 2.0), std::invalid_argument);
  }

  SECTION("setting a coefficient to zero removes the term") {
    auto p = MP2({
        {{0, 0}, 1},
        {{1, 0}, 2},
    });

    p.set(Eigen::Array2i({1, 0}), 0.0);

    REQUIRE(p.size() == 1);
    REQUIRE(p == MP2(1));
    REQUIRE_FALSE(p.contains(Eigen::Array2i({1, 0})));
    REQUIRE(p.get(Eigen::Array2i::Zero()) == 1);
  }

  SECTION("scalar addition and subtraction preserve canonical zero") {
    auto p = MP2(2.0);

    p += -2.0;
    require_canonical_zero(p);

    p -= 0.0;
    require_canonical_zero(p);
  }

  SECTION("polynomial addition preserves canonical zero after cancellation") {
    auto p = MP2({
        {{0, 0}, 1},
        {{1, 0}, 2},
    });
    auto q = MP2({
        {{0, 0}, -1},
        {{1, 0}, -2},
    });

    p += q;

    require_canonical_zero(p);
  }

  SECTION("polynomial subtraction preserves canonical zero after cancellation") {
    auto p = MP2({
        {{0, 0}, 1},
        {{0, 1}, 3},
    });
    auto q = MP2({
        {{0, 0}, 1},
        {{0, 1}, 3},
    });

    p -= q;

    require_canonical_zero(p);
  }
}

TEST_CASE("pow", "[mvPolynomial]") {
  auto m = MP2({
      {{0, 0}, 1},
      {{1, 0}, 2},
      {{0, 1}, 3},
  });

  auto m3 = m * m * m;

  SECTION("0") { REQUIRE(m.pow(0) == MP2(1)); }

  SECTION("1") { REQUIRE(m.pow(1) == m); }

  SECTION("2") { REQUIRE(m.pow(2) == m * m); }

  SECTION("3") { REQUIRE(m.pow(3) == m * m * m); }

  SECTION("7") { REQUIRE(m.pow(7) == m3 * m3 * m); }

  SECTION("15") { REQUIRE(m.pow(15) == m3 * m3 * m3 * m3 * m3); }
}

TEST_CASE("allocator propagation", "[mvPolynomial]") {
  auto allocations = std::make_shared<std::size_t>(0);
  auto alloc       = CountingPairAllocator(allocations);
  auto p           = CountingMP2(
      {
          {{0, 0}, 1},
          {{1, 0}, 2},
          {{0, 1}, 3},
  },
      alloc
  );

  SECTION("pow uses rebound allocator") {
    const auto before = *allocations;
    auto       p7     = p.pow(7);

    REQUIRE(*allocations > before);
    REQUIRE(p7 == p.pow(7));
  }

  SECTION("point evaluation uses rebound allocator") {
    const auto before = *allocations;
    auto       value  = p(Eigen::Vector2d({2, 3}));

    REQUIRE(*allocations > before);
    REQUIRE(value == 14);
  }

  SECTION("polynomial composition uses rebound allocator") {
    auto x = CountingMP2(
        {
            {{0, 0}, 1},
            {{1, 0}, 2},
            {{0, 1}, 3},
    },
        alloc
    );

    const auto before   = *allocations;
    auto       composed = p(x, 1);

    REQUIRE(*allocations > before);
    REQUIRE(
        composed == CountingMP2(
                        {
                            {{0, 0}, 4},
                            {{1, 0}, 8},
                            {{0, 1}, 9},
    },
                        alloc
                    )
    );
  }
}

TEST_CASE("operator()", "[mvPolynomial]") {
  SECTION("point") {
    auto m = MP2({
        {{0, 0}, 1},
        {{1, 0}, 2},
        {{0, 1}, 3},
        {{1, 1}, 4},
        {{2, 0}, 5},
        {{0, 2}, 6},
    });

    REQUIRE(m(Eigen::Vector2d::Zero()) == 1);
    REQUIRE(m(Eigen::Vector2d({2, 3})) == 112);
  }

  SECTION("point without constant") {
    auto m = MP2({
        {{1, 0}, 2},
        {{0, 1}, 3},
        {{1, 1}, 4},
        {{2, 0}, 5},
        {{0, 2}, 6},
    });

    REQUIRE(m(Eigen::Vector2d::Zero()) == 0);
    REQUIRE(m(Eigen::Vector2d({2, 3})) == 111);
  }

  SECTION("mvpolynomial_1d") {
    auto m = MP3({
        {{0, 0, 0}, 1},
        {{1, 0, 0}, 2},
        {{0, 1, 0}, 3},
        {{0, 0, 1}, 4},
    });
    auto x = MP3({
        {{0, 0, 0}, 1},
        {{1, 0, 0}, 2},
        {{0, 1, 0}, 3},
    });
    REQUIRE(
        m(x, 2) == MP3({
                       {{0, 0, 0},  5},
                       {{1, 0, 0}, 10},
                       {{0, 1, 0}, 15},
    })
    );
    auto y = MP3({
        {{0, 0, 0}, 1},
        {{1, 0, 0}, 2},
        {{0, 0, 1}, 4},
    });
    REQUIRE(
        m(y, 1) == MP3({
                       {{0, 0, 0},  4},
                       {{1, 0, 0},  8},
                       {{0, 0, 1}, 16},
    })
    );
  }

  SECTION("mvpolynomial_1d_without_constant") {
    auto m = MP3({
        {{1, 0, 0}, 2},
        {{0, 1, 0}, 3},
        {{0, 0, 1}, 4},
    });
    auto x = MP3({
        {{0, 0, 0}, 1},
        {{1, 0, 0}, 2},
        {{0, 1, 0}, 3},
    });
    REQUIRE(
        m(x, 2) == MP3({
                       {{0, 0, 0},  4},
                       {{1, 0, 0}, 10},
                       {{0, 1, 0}, 15},
    })
    );
  }

  SECTION("mvpolynomial_2d") {
    auto m = MP3({
        {{0, 0, 0}, 1},
        {{2, 0, 0}, 2},
        {{0, 2, 0}, 3},
        {{0, 0, 2}, 4},
    });
    auto x = MP3({
        {{0, 0, 0}, 1},
        {{1, 0, 0}, 2},
        {{0, 1, 0}, 3},
    });
    REQUIRE(
        m(x, 2) == MP3({
                       {{0, 0, 0},  5},
                       {{1, 0, 0}, 16},
                       {{0, 1, 0}, 24},
                       {{1, 1, 0}, 48},
                       {{2, 0, 0}, 18},
                       {{0, 2, 0}, 39},
    })
    );
    auto y = MP3({
        {{0, 0, 0}, 1},
        {{1, 0, 0}, 2},
        {{0, 0, 1}, 4},
    });
    REQUIRE(
        m(y, 1) == MP3({
                       {{0, 0, 0},  4},
                       {{1, 0, 0}, 12},
                       {{0, 0, 1}, 24},
                       {{1, 0, 1}, 48},
                       {{2, 0, 0}, 14},
                       {{0, 0, 2}, 52},
    })
    );
  }

  SECTION("point_randomized_matches_naive_sum") {
    auto rng             = std::mt19937(20260517);
    auto term_count_dist = std::uniform_int_distribution<int>(1, 12);

    for (int trial = 0; trial < 200; ++trial) {
      auto m        = MakeRandomPolynomial<MP3>(rng, term_count_dist(rng), 4, 5);
      auto x        = MakeRandomPoint<MP3>(rng, -3, 3);
      auto actual   = m(x);
      auto expected = NaivePointEval(m, x);

      CAPTURE(trial, DescribePolynomial(m), DescribePoint<MP3>(x));
      REQUIRE(actual == expected);
    }
  }

  SECTION("composition_randomized_matches_naive_sum") {
    auto rng                   = std::mt19937(20260518);
    auto outer_term_count_dist = std::uniform_int_distribution<int>(1, 10);
    auto inner_term_count_dist = std::uniform_int_distribution<int>(1, 6);
    auto axis_dist             = std::uniform_int_distribution<int>(0, MP3::dim - 1);

    for (int trial = 0; trial < 120; ++trial) {
      auto outer    = MakeRandomPolynomial<MP3>(rng, outer_term_count_dist(rng), 3, 4);
      auto inner    = MakeRandomPolynomial<MP3>(rng, inner_term_count_dist(rng), 2, 3);
      auto axis     = axis_dist(rng);
      auto actual   = outer(inner, axis);
      auto expected = NaiveCompose(outer, inner, axis);

      CAPTURE(
          trial,
          axis,
          DescribePolynomial(outer),
          DescribePolynomial(inner),
          DescribePolynomial(actual),
          DescribePolynomial(expected)
      );
      REQUIRE(actual == expected);
    }
  }
}

TEST_CASE("D", "[mvPolynomial]") {
  auto m = MP2({
      {{0, 0}, 1},
      {{1, 0}, 2},
      {{0, 1}, 3},
      {{1, 1}, 4},
      {{2, 0}, 5},
      {{0, 2}, 6},
  });

  SECTION("D_x") {
    auto ans = std::vector<std::pair<Eigen::Array2i, double>>({
        {{0, 0},  2},
        {{0, 1},  4},
        {{1, 0}, 10}
    });
    auto dm0 = mvPolynomial::D(m, 0);

    REQUIRE(dm0.size() == ans.size());
    for (size_t i = 0; i < ans.size(); ++i) {
      REQUIRE(dm0.get(ans[i].first) == ans[i].second);
    }
  }

  SECTION("D_y") {
    auto ans = std::vector<std::pair<Eigen::Array2i, double>>({
        {{0, 0},  3},
        {{1, 0},  4},
        {{0, 1}, 12}
    });
    auto dm1 = mvPolynomial::D(m, 1);

    REQUIRE(dm1.size() == ans.size());
    for (size_t i = 0; i < ans.size(); ++i) {
      REQUIRE(dm1.get(ans[i].first) == ans[i].second);
    }
  }

  SECTION("zero result stays canonical zero polynomial") {
    auto constant = MP2(7.0);
    auto dx       = mvPolynomial::D(constant, 0);

    REQUIRE(dx.size() == 1);
    REQUIRE(dx == MP2());
    REQUIRE(dx.contains(Eigen::Array2i::Zero()));
    REQUIRE(dx.get(Eigen::Array2i::Zero()) == 0);
  }
}

TEST_CASE("integral", "[mvPolynomial]") {
  auto m = MP2({
      {{0, 0}, 1},
      {{1, 0}, 2},
      {{0, 1}, 3},
      {{1, 1}, 4},
      {{2, 0}, 5},
      {{0, 2}, 6},
  });

  SECTION("S m dx") {
    auto ans = std::vector<std::pair<Eigen::Array2i, double>>({
        {{1, 0},       1},
        {{2, 0},       1},
        {{1, 1},       3},
        {{2, 1},       2},
        {{3, 0}, 5.0 / 3},
        {{1, 2},       6}
    });
    auto sm  = mvPolynomial::Integrate(m, 0);

    REQUIRE(sm.size() == ans.size());
    for (size_t i = 0; i < ans.size(); ++i) {
      REQUIRE(sm.get(ans[i].first) == ans[i].second);
    }
  }

  SECTION("S m dy") {
    auto ans = std::vector<std::pair<Eigen::Array2i, double>>({
        {{0, 1},   1},
        {{1, 1},   2},
        {{0, 2}, 1.5},
        {{1, 2},   2},
        {{2, 1},   5},
        {{0, 3},   2}
    });
    auto sm  = mvPolynomial::Integrate(m, 1);

    REQUIRE(sm.size() == ans.size());
    for (size_t i = 0; i < ans.size(); ++i) {
      REQUIRE(sm.get(ans[i].first) == ans[i].second);
    }
  }

  SECTION("constant term preserves searchable ordering") {
    auto c  = MP2({
        {{0, 0}, 9},
    });
    auto sc = mvPolynomial::Integrate(c, 1);

    REQUIRE(sc.size() == 1);
    REQUIRE(sc.contains(Eigen::Array2i({0, 1})));
    REQUIRE(sc.find(Eigen::Array2i({0, 1})) != sc.end());
    REQUIRE(sc.lower_bound(Eigen::Array2i({0, 1})) != sc.end());
    REQUIRE((sc.lower_bound(Eigen::Array2i({0, 1}))->first == Eigen::Array2i({0, 1})).all());
    REQUIRE(sc.get(Eigen::Array2i({0, 1})) == 9);
  }

  SECTION("higher degree term keeps lookup behavior") {
    auto high = MP2({
        {{3, 2}, 8},
    });
    auto shi  = mvPolynomial::Integrate(high, 0);

    REQUIRE(shi.size() == 1);
    REQUIRE(shi.contains(Eigen::Array2i({4, 2})));
    REQUIRE(shi.find(Eigen::Array2i({4, 2})) != shi.end());
    REQUIRE(shi.lower_bound(Eigen::Array2i({4, 2})) != shi.end());
    REQUIRE((shi.lower_bound(Eigen::Array2i({4, 2}))->first == Eigen::Array2i({4, 2})).all());
    REQUIRE(shi.get(Eigen::Array2i({4, 2})) == 2);
  }

  SECTION("zero polynomial stays canonical zero after integration") {
    auto zero = MP2();
    auto sz   = mvPolynomial::Integrate(zero, 0);

    REQUIRE(sz.size() == 1);
    REQUIRE(sz == MP2());
    REQUIRE(sz.contains(Eigen::Array2i::Zero()));
    REQUIRE(sz.get(Eigen::Array2i::Zero()) == 0);
  }
}

TEST_CASE("multiply", "[mvPolynomial]") {
  // left hand
  auto l = MP2({
      {{0, 0}, 1},
      {{1, 0}, 2},
      {{0, 1}, 3},
  });

  // right hand
  auto r = MP2({
      {{2, 0},  5},
      {{0, 2},  7},
      {{1, 1}, 11},
  });

  auto ans = std::vector<std::pair<Eigen::Array2i, double>>({
      {{2, 0},  5},
      {{3, 0}, 10},
      {{2, 1}, 37}, // {{2, 1}, 15} + {{2, 1}, 22}
      {{0, 2},  7},
      {{1, 2}, 47}, // {{1, 2}, 14} + {{1, 2}, 33}
      {{0, 3}, 21},
      {{1, 1}, 11},
  });

  auto prod = l * r;

  REQUIRE(prod.size() == ans.size());
  for (size_t i = 0; i < ans.size(); ++i) {
    REQUIRE(prod.get(ans[i].first) == ans[i].second);
  }

  SECTION("multiply assign by monomial matches binary multiply") {
    auto lhs      = MP2({
        {{1, 0}, 2},
        {{0, 1}, 3},
    });
    auto monomial = MP2({
        {{2, 1}, 5},
    });
    auto expected = MP2({
        {{3, 1}, 10},
        {{2, 2}, 15},
    });

    auto inplace = lhs;
    inplace *= monomial;

    REQUIRE(inplace == expected);
    REQUIRE(inplace == lhs * monomial);
    REQUIRE(inplace.contains(Eigen::Array2i({3, 1})));
    REQUIRE(inplace.find(Eigen::Array2i({2, 2})) != inplace.end());
    REQUIRE(inplace.lower_bound(Eigen::Array2i({2, 2})) != inplace.end());
    REQUIRE((inplace.lower_bound(Eigen::Array2i({2, 2}))->first == Eigen::Array2i({2, 2})).all());
  }

  SECTION("multiplication normalizes zero coefficient collisions") {
    auto cancel_l = MP2({
        {{0, 0},  1},
        {{1, 0}, -1},
    });
    auto cancel_r = MP2({
        {{0, 0}, 1},
        {{1, 0}, 1},
    });

    auto cancel_prod = cancel_l * cancel_r;

    REQUIRE(cancel_prod.size() == 2);
    REQUIRE(cancel_prod.get(Eigen::Array2i({0, 0})) == 1);
    REQUIRE_FALSE(cancel_prod.contains(Eigen::Array2i({1, 0})));
    REQUIRE(cancel_prod.get(Eigen::Array2i({2, 0})) == -1);
  }
}

TEST_CASE("sum", "[mvPolynomial]") {
  auto l = MP2({
      {{0, 0}, 1},
      {{1, 0}, 2},
      {{0, 1}, 3},
  });

  auto r = MP2({
      {{2, 0},  5},
      {{0, 2},  7},
      {{1, 1}, 11},
  });

  auto ans = std::vector<std::pair<Eigen::Array2i, double>>({
      {{0, 0},  1},
      {{1, 0},  2},
      {{0, 1},  3},
      {{2, 0},  5},
      {{0, 2},  7},
      {{1, 1}, 11},
  });
  auto sum = l + r;

  REQUIRE(sum.size() == ans.size());
  for (size_t i = 0; i < ans.size(); ++i) {
    REQUIRE(sum.get(ans[i].first) == ans[i].second);
  }
}

TEST_CASE("sub", "[mvPolynomial]") {
  auto l   = MP2({
      {{0, 0}, 1},
      {{1, 0}, 2},
      {{0, 1}, 3},
  });
  auto r   = MP2({
      {{2, 0},  5},
      {{0, 2},  7},
      {{1, 1}, 11},
  });
  auto ans = std::vector<std::pair<Eigen::Array2i, double>>({
      {{0, 0},   1},
      {{1, 0},   2},
      {{0, 1},   3},
      {{2, 0},  -5},
      {{0, 2},  -7},
      {{1, 1}, -11},
  });
  auto sub = l - r;

  REQUIRE(sub.size() == ans.size());
  for (size_t i = 0; i < ans.size(); ++i) {
    REQUIRE(sub.get(ans[i].first) == ans[i].second);
  }
}
