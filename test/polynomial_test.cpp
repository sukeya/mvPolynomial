#define BOOST_TEST_MODULE mvPolynomial_unit_test

#include "boost/test/unit_test.hpp"
#include "mvPolynomial/polynomial.hpp"

#include <vector>

namespace utf = boost::unit_test;
namespace tt  = boost::test_tools;

using Poly = mvPolynomial::Polynomial<int, double>;

BOOST_AUTO_TEST_CASE(polynomial_init, *utf::tolerance(tt::fpc::percent_tolerance(1e-10))) {
  auto ans = std::vector<std::pair<int, double>>();
  ans      = {
      {0, 1},
      {1, 2},
      {2, 3}
  };

  auto m = Poly(ans.begin(), ans.end());
  for (auto i = 0; i < ans.size(); ++i) {
    BOOST_TEST(m[ans[i].first] == ans[i].second);
  }
}

BOOST_AUTO_TEST_CASE(polynomial_Of, *utf::tolerance(tt::fpc::percent_tolerance(1e-10))) {
  auto ans = std::vector<std::pair<int, double>>();
  ans      = {
      {0, 1},
      {1, 2},
      {2, 3}
  };

  auto m = Poly(ans.begin(), ans.end());
  BOOST_TEST(Of(m, 0) == 1);
  BOOST_TEST(Of(m, 2) == 17);
}

BOOST_AUTO_TEST_CASE(polynomial_derivative, *utf::tolerance(tt::fpc::percent_tolerance(1e-10))) {
  auto ans = std::vector<std::pair<int, double>>();
  ans      = {
      {0, 1},
      {1, 2},
      {2, 3}
  };
  auto m = Poly(ans.begin(), ans.end());
  ans    = {
      {0, 2},
      {1, 6}
  };
  auto dm0 = mvPolynomial::D(m);
  for (auto i = 0; i < ans.size(); ++i) {
    BOOST_TEST(dm0[ans[i].first] == ans[i].second);
  }
}

BOOST_AUTO_TEST_CASE(polynomial_integral, *utf::tolerance(tt::fpc::percent_tolerance(1e-10))) {
  auto ans = std::vector<std::pair<int, double>>();
  ans      = {
      {0, 1},
      {1, 2},
      {2, 3}
  };
  auto m = Poly(ans.begin(), ans.end());
  ans    = {
      {1, 1},
      {2, 1},
      {3, 1}
  };
  auto sm = mvPolynomial::Integrate(m);
  for (auto i = 0; i < ans.size(); ++i) {
    BOOST_TEST(sm[ans[i].first] == ans[i].second);
  }
}
