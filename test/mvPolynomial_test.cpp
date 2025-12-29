#include "mvPolynomial/mvPolynomial.hpp"

#include <cstddef>
#include <vector>

#include <catch2/catch_test_macros.hpp>

using MP2 = mvPolynomial::MVPolynomial<int, double, 2>;
using MP3 = mvPolynomial::MVPolynomial<int, double, 3>;

TEST_CASE("constructor", "[mvPolynomial]") {
  SECTION("default") {
    auto ans = std::vector<std::pair<Eigen::Array2i, double>>({
        {{0, 0}, 0},
    });
    auto m   = MP2();

    REQUIRE(m.size() == ans.size());
    for (auto i = 0; i < ans.size(); ++i) {
      REQUIRE(m[ans[i].first] == ans[i].second);
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
      REQUIRE(m.at(ans[i].first) == ans[i].second);
    }
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
      REQUIRE(m.at(ans[i].first) == ans[i].second);
    }
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
        m(x, 2)
        == MP3({
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
        m(y, 1)
        == MP3({
            {{0, 0, 0},  4},
            {{1, 0, 0},  8},
            {{0, 0, 1}, 16},
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
        m(x, 2)
        == MP3({
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
        m(y, 1)
        == MP3({
            {{0, 0, 0},  4},
            {{1, 0, 0}, 12},
            {{0, 0, 1}, 24},
            {{1, 0, 1}, 48},
            {{2, 0, 0}, 14},
            {{0, 0, 2}, 52},
    })
    );
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
      REQUIRE(dm0[ans[i].first] == ans[i].second);
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
      REQUIRE(dm1[ans[i].first] == ans[i].second);
    }
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
      REQUIRE(sm[ans[i].first] == ans[i].second);
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
      REQUIRE(sm[ans[i].first] == ans[i].second);
    }
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
    REQUIRE(prod[ans[i].first] == ans[i].second);
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
    REQUIRE(sum[ans[i].first] == ans[i].second);
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
  for (auto i = 0; i < ans.size(); ++i) {
    REQUIRE(sub[ans[i].first] == ans[i].second);
  }
}
