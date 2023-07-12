#define BOOST_TEST_MODULE multivar_polynomial_unit_test


#include "boost/test/unit_test.hpp"
#include "multivar_polynomial/multivar_polynomial.hpp"

#include <vector>


namespace utf = boost::unit_test;
namespace tt = boost::test_tools;

using MP2 = multivar_polynomial::MultiVarPolynomial<int, double, 2>;
using MP3 = multivar_polynomial::MultiVarPolynomial<int, double, 3>;

using EO3 = multivar_polynomial::ExactOf<int, double, 3>;


BOOST_AUTO_TEST_CASE(multivar_polynomial_init, * utf::tolerance(tt::fpc::percent_tolerance(1e-10)))
{
  auto ans = std::vector<std::pair<Eigen::Array2i, double>>();
  ans = {
    {{0, 0}, 1},
    {{1, 0}, 2},
    {{0, 1}, 3},
    {{1, 1}, 4},
    {{2, 0}, 5},
    {{0, 2}, 6},
  };

  auto m = MP2(ans.begin(), ans.end());
  for (auto i = 0; i < ans.size(); ++i)
  {
    BOOST_TEST(m[ans[i].first] == ans[i].second);
  }
}


BOOST_AUTO_TEST_CASE(multivar_polynomial_Of, * utf::tolerance(tt::fpc::percent_tolerance(1e-10)))
{
  auto ans = std::vector<std::pair<Eigen::Array2i, double>>();
  ans = {
    {{0, 0}, 1},
    {{1, 0}, 2},
    {{0, 1}, 3},
    {{1, 1}, 4},
    {{2, 0}, 5},
    {{0, 2}, 6},
  };

  auto m = MP2(ans.begin(), ans.end());
  BOOST_TEST(Of(m, Eigen::Vector2d::Zero()) == 1);
  BOOST_TEST(Of(m, {2, 3}) == 112);
}


BOOST_AUTO_TEST_CASE(multivar_polynomial_derivative, * utf::tolerance(tt::fpc::percent_tolerance(1e-10)))
{
  auto ans = std::vector<std::pair<Eigen::Array2i, double>>();
  ans = {
    {{0, 0}, 1},
    {{1, 0}, 2},
    {{0, 1}, 3},
    {{1, 1}, 4},
    {{2, 0}, 5},
    {{0, 2}, 6},
  };
  auto m = MP2(ans.begin(), ans.end());
  ans = {
    {{0, 0}, 2},
    {{0, 1}, 4},
    {{1, 0}, 10}
  };
  auto dm0 = multivar_polynomial::D(m, 0);
  for (auto i = 0; i < ans.size(); ++i)
  {
    BOOST_TEST(dm0[ans[i].first] == ans[i].second);
  }
  ans = {
    {{0, 0}, 3},
    {{1, 0}, 4},
    {{0, 1}, 12}
  };
  auto dm1 = multivar_polynomial::D(m, 1);
  for (auto i = 0; i < ans.size(); ++i)
  {
    BOOST_TEST(dm1[ans[i].first] == ans[i].second);
  }
}


BOOST_AUTO_TEST_CASE(multivar_polynomial_integral, * utf::tolerance(tt::fpc::percent_tolerance(1e-10)))
{
  auto ans = std::vector<std::pair<Eigen::Array2i, double>>();
  ans = {
    {{0, 0}, 1},
    {{1, 0}, 2},
    {{0, 1}, 3},
    {{1, 1}, 4},
    {{2, 0}, 5},
    {{0, 2}, 6},
  };
  auto m = MP2(ans.begin(), ans.end());
  ans = {
    {{0, 1}, 1},
    {{1, 1}, 2},
    {{0, 2}, 1.5},
    {{1, 2}, 2},
    {{2, 1}, 5},
    {{0, 3}, 2}
  };
  auto sm = multivar_polynomial::Integrate(m, 1);
  for (auto i = 0; i < ans.size(); ++i)
  {
    BOOST_TEST(sm[ans[i].first] == ans[i].second);
  }
}


BOOST_AUTO_TEST_CASE(multivar_polynomial_multiply, * utf::tolerance(tt::fpc::percent_tolerance(1e-10)))
{
  auto ans = std::vector<std::pair<Eigen::Array2i, double>>();
  ans = {
    {{0, 0}, 1},
    {{1, 0}, 2},
    {{0, 1}, 3},
  };
  auto l = MP2(ans.begin(), ans.end());
  ans = {
    {{2, 0}, 5},
    {{0, 2}, 7},
  };
  auto m = MP2(ans.begin(), ans.end());
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
    BOOST_TEST(prod[ans[i].first] == ans[i].second);
  }
}


BOOST_AUTO_TEST_CASE(multivar_polynomial_sum, * utf::tolerance(tt::fpc::percent_tolerance(1e-10)))
{
  auto ans = std::vector<std::pair<Eigen::Array2i, double>>();
  ans = {
    {{0, 0}, 1},
    {{1, 0}, 2},
    {{0, 1}, 3},
  };
  auto l = MP2(ans.begin(), ans.end());
  ans = {
    {{2, 0}, 5},
    {{0, 2}, 7},
  };
  auto m = MP2(ans.begin(), ans.end());
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
    BOOST_TEST(sum[ans[i].first] == ans[i].second);
  }
}


BOOST_AUTO_TEST_CASE(multivar_polynomial_sub, * utf::tolerance(tt::fpc::percent_tolerance(1e-10)))
{
  auto ans = std::vector<std::pair<Eigen::Array2i, double>>();
  ans = {
    {{0, 0}, 1},
    {{1, 0}, 2},
    {{0, 1}, 3},
  };
  auto l = MP2(ans.begin(), ans.end());
  ans = {
    {{2, 0}, 5},
    {{0, 2}, 7},
  };
  auto m = MP2(ans.begin(), ans.end());
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
    BOOST_TEST(sub[ans[i].first] == ans[i].second);
  }
}


BOOST_AUTO_TEST_CASE(multivar_polynomial_default_value_check, * utf::tolerance(tt::fpc::percent_tolerance(1e-10)))
{
  auto ans = std::vector<std::pair<Eigen::Array2i, double>>();
  ans = {
    {{0, 0}, 0},
  };
  auto m = MP2();
  for (auto i = 0; i < ans.size(); ++i)
  {
    BOOST_TEST(m[ans[i].first] == ans[i].second);
  }
}


BOOST_AUTO_TEST_CASE(multivar_polynomial_exact_of_init, * utf::tolerance(tt::fpc::percent_tolerance(1e-10)))
{
  auto ans = std::vector<std::pair<Eigen::Vector3i, double>>();
  ans = {
    {{0, 0, 0}, 1},
    {{1, 0, 0}, 2},
    {{0, 1, 0}, 3},
    {{0, 0, 1}, 4},
    {{2, 0, 0}, 5},
    {{0, 2, 0}, 6},
    {{0, 0, 2}, 7},
    {{1, 1, 0}, 8},
    {{0, 1, 1}, 9},
    {{1, 0, 1}, 10},
  };
  auto m = MP3(ans.begin(), ans.end());
  auto exact_of = EO3(m);
  for (auto i = 0; i < ans.size(); ++i)
  {
    const auto& p = exact_of.get_polynomial();
    BOOST_TEST(p[ans[i].first] == ans[i].second);
  }
  /*
  auto ans2 = std::vector<std::pair<Eigen::Array2i, double>>();
  ans2 = {
    {{0, 0}, 0},
    {{1, 0}, 0},
    {{0, 1}, 0},
    {{2, 0}, 0},
    {{0, 2}, 0},
    {{1, 1}, 0},
  };
  for (auto i = 0; i < ans2.size(); ++i)
  {
    const auto& p = exact_of.projection_.get_polynomial();
    BOOST_TEST(p[ans2[i].first] == 0);
  }
  auto ans3 = std::vector<std::pair<int, double>>();
  ans3 = {
    {0, 0},
    {1, 0},
    {2, 0}
  };
  for (auto i = 0; i < ans3.size(); ++i)
  {
    const auto& p = exact_of.projection_.projection_.get_polynomial();
    BOOST_TEST(p[ans3[i].first] == 0);
  }
  */
  BOOST_TEST(exact_of({0, 0, 0}) == 1);
  BOOST_TEST(exact_of({1, 0, 0}) == 1 + 2 + 5);
  BOOST_TEST(exact_of({2, 0, 0}) == 1 + 2 * 2 + 5 * 4);
  BOOST_TEST(exact_of({0, 1, 0}) == 1 + 3 + 6);
  BOOST_TEST(exact_of({0, 2, 0}) == 1 + 3 * 2 + 6 * 4);
  BOOST_TEST(exact_of({0, 0, 1}) == 1 + 4 + 7);
  BOOST_TEST(exact_of({0, 0, 2}) == 1 + 4 * 2 + 7 * 4);
  BOOST_TEST(exact_of({1, 1, 0}) == 1 + 2 + 5 + 3 + 6 + 8);
  BOOST_TEST(exact_of({0, 1, 1}) == 1 + 3 + 6 + 4 + 7 + 9);
  BOOST_TEST(exact_of({1, 0, 1}) == 1 + 2 + 5 + 4 + 7 + 10);
}