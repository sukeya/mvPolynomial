#define BOOST_TEST_MODULE multivar_polynomial_unit_test


#include "boost/test/unit_test.hpp"
#include "multivar_polynomial/multivar_polynomial.hpp"

#include <iostream>
#include <vector>


namespace utf = boost::unit_test;
namespace tt = boost::test_tools;

BOOST_AUTO_TEST_CASE(multivar_polynomial_init, * utf::tolerance(tt::fpc::percent_tolerance(1e-10)))
{
  auto ans = std::vector<std::pair<Eigen::Vector2i, double>>();
  ans = {
    {{0, 0}, 1},
    {{1, 0}, 1},
    {{0, 1}, 1},
    {{1, 1}, 1},
    {{2, 0}, 1},
    {{0, 2}, 1},
  };

  auto m = multivar_polynomial::MultiVarPolynomial<double, int, 2>(ans.begin(), ans.end());
  for (auto i = 0; i < ans.size(); ++i)
  {
    BOOST_TEST(m[ans[i].first] == ans[i].second);
  }
}

BOOST_AUTO_TEST_CASE(multivar_polynomial_Of, * utf::tolerance(tt::fpc::percent_tolerance(1e-10)))
{
  auto ans = std::vector<std::pair<Eigen::Vector2i, double>>();
  ans = {
    {{0, 0}, 1},
    {{1, 0}, 1},
    {{0, 1}, 1},
    {{1, 1}, 1},
    {{2, 0}, 1},
    {{0, 2}, 1},
  };

  auto m = multivar_polynomial::MultiVarPolynomial<double, int, 2>(ans.begin(), ans.end());
  assert(m.Of(Eigen::Vector2d::Zero()) == 1);
  assert(m.Of({2, 3}) == 25);
}

BOOST_AUTO_TEST_CASE(multivar_polynomial_derivative, * utf::tolerance(tt::fpc::percent_tolerance(1e-10)))
{
  auto ans = std::vector<std::pair<Eigen::Vector2i, double>>();
  ans = {
    {{0, 0}, 1},
    {{1, 0}, 1},
    {{0, 1}, 1},
    {{1, 1}, 1},
    {{2, 0}, 1},
    {{0, 2}, 1},
  };
  auto m = multivar_polynomial::MultiVarPolynomial<double, int, 2>(ans.begin(), ans.end());
  ans = {
    {{0, 0}, 1},
    {{0, 1}, 1},
    {{1, 0}, 2}
  };
  auto dm = multivar_polynomial::Derivative(m, 0);
  for (auto i = 0; i < ans.size(); ++i)
  {
    assert(dm[ans[i].first] == ans[i].second);
  }
}

BOOST_AUTO_TEST_CASE(multivar_polynomial_integral, * utf::tolerance(tt::fpc::percent_tolerance(1e-10)))
{
  auto ans = std::vector<std::pair<Eigen::Vector2i, double>>();
  ans = {
    {{0, 0}, 1},
    {{1, 0}, 1},
    {{0, 1}, 1},
    {{1, 1}, 1},
    {{2, 0}, 1},
    {{0, 2}, 1},
  };
  auto m = multivar_polynomial::MultiVarPolynomial<double, int, 2>(ans.begin(), ans.end());
  ans = {
    {{0, 1}, 1},
    {{1, 1}, 1},
    {{0, 2}, 0.5},
    {{1, 2}, 0.5},
    {{2, 1}, 1},
    {{0, 3}, 1.0 / 3}
  };
  auto sm = multivar_polynomial::Integral(m, 1);
  for (auto i = 0; i < ans.size(); ++i)
  {
    assert(sm[ans[i].first] == ans[i].second);
  }
}

BOOST_AUTO_TEST_CASE(multivar_polynomial_multiply, * utf::tolerance(tt::fpc::percent_tolerance(1e-10)))
{
  auto ans = std::vector<std::pair<Eigen::Vector2i, double>>();
  ans = {
    {{0, 0}, 1},
    {{1, 0}, 2},
    {{0, 1}, 3},
  };
  auto l = multivar_polynomial::MultiVarPolynomial<double, int, 2>(ans.begin(), ans.end());
  ans = {
    {{2, 0}, 5},
    {{0, 2}, 7},
  };
  auto m = multivar_polynomial::MultiVarPolynomial<double, int, 2>(ans.begin(), ans.end());
  ans = {
    {{2, 0}, 5},
    {{3, 0}, 10},
    {{2, 1}, 15},
    {{0, 2}, 7},
    {{1, 2}, 14},
    {{0, 3}, 21},
  };
  auto prod = l * m;
  for (auto i = 0; i < ans.size(); ++i)
  {
    assert(prod[ans[i].first] == ans[i].second);
  }
}

BOOST_AUTO_TEST_CASE(multivar_polynomial_sum, * utf::tolerance(tt::fpc::percent_tolerance(1e-10)))
{
  auto ans = std::vector<std::pair<Eigen::Vector2i, double>>();
  ans = {
    {{0, 0}, 1},
    {{1, 0}, 2},
    {{0, 1}, 3},
  };
  auto l = multivar_polynomial::MultiVarPolynomial<double, int, 2>(ans.begin(), ans.end());
  ans = {
    {{2, 0}, 5},
    {{0, 2}, 7},
  };
  auto m = multivar_polynomial::MultiVarPolynomial<double, int, 2>(ans.begin(), ans.end());
  ans = {
    {{0, 0}, 1},
    {{1, 0}, 2},
    {{0, 1}, 3},
    {{2, 0}, 5},
    {{0, 2}, 7},
  };
  auto sum = l + m;
  for (auto i = 0; i < ans.size(); ++i)
  {
    assert(sum[ans[i].first] == ans[i].second);
  }
}

BOOST_AUTO_TEST_CASE(multivar_polynomial_sub, * utf::tolerance(tt::fpc::percent_tolerance(1e-10)))
{
  auto ans = std::vector<std::pair<Eigen::Vector2i, double>>();
  ans = {
    {{0, 0}, 1},
    {{1, 0}, 2},
    {{0, 1}, 3},
  };
  auto l = multivar_polynomial::MultiVarPolynomial<double, int, 2>(ans.begin(), ans.end());
  ans = {
    {{2, 0}, 5},
    {{0, 2}, 7},
  };
  auto m = multivar_polynomial::MultiVarPolynomial<double, int, 2>(ans.begin(), ans.end());
  ans = {
    {{0, 0}, 1},
    {{1, 0}, 2},
    {{0, 1}, 3},
    {{2, 0}, -5},
    {{0, 2}, -7},
  };
  auto sub = l - m;
  for (auto i = 0; i < ans.size(); ++i)
  {
    assert(sub[ans[i].first] == ans[i].second);
  }
}