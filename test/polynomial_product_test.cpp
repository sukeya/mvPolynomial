#define BOOST_TEST_MODULE polynomial_product_unit_test


#include "boost/test/unit_test.hpp"
#include "multivar_polynomial/polynomial_product.hpp"

#include <vector>


namespace utf = boost::unit_test;
namespace tt = boost::test_tools;


BOOST_AUTO_TEST_CASE(polynomial_product_init, * utf::tolerance(tt::fpc::percent_tolerance(1e-10)))
{
  using Poly = typename multivar_polynomial::PolynomialProduct<double, 2>::polynomial_type;
  auto v = std::array<std::vector<double>, 2>({{1, 2, 3}, {4, 5}});
  auto pp = multivar_polynomial::PolynomialProduct<double, 2>({{1, 2, 3}, {4, 5}});
  for (std::size_t i = 0; i != v.size(); ++i)
  {
    BOOST_TEST(Poly(v[i]) == pp[i]);
  }
}

