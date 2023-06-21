#define BOOST_TEST_MODULE polynomial_product_unit_test


#include "boost/test/unit_test.hpp"
#include "multivar_polynomial/polynomial_product.hpp"
#include "multivar_polynomial/polynomial.hpp"

#include <vector>


namespace utf = boost::unit_test;
namespace tt = boost::test_tools;


BOOST_AUTO_TEST_CASE(polynomial_product_init, * utf::tolerance(tt::fpc::percent_tolerance(1e-10)))
{
  using Poly = multivar_polynomial::Polynomial<double, int>;
  auto v = std::array<Poly, 2>({
    Poly({
      {0, 1},
      {1, 2},
      {2, 3}
    }),
    Poly({
      {0, 4},
      {1, 5}
    })
  });
  auto pp = multivar_polynomial::PolynomialProduct<Poly, 2>(v.begin(), v.end());
  BOOST_TEST(v.size() == pp.size());
  for (std::size_t i = 0; i != v.size(); ++i)
  {
    BOOST_TEST(v[i] == pp[i]);
  }
}


BOOST_AUTO_TEST_CASE(polynomial_product_Of, * utf::tolerance(tt::fpc::percent_tolerance(1e-10)))
{
  using Poly = multivar_polynomial::Polynomial<double, int>;
  auto v = std::array<Poly, 2>({
    Poly({
      {0, 1},
      {1, 2},
      {2, 3}
    }),
    Poly({
      {0, 4},
      {1, 5}
    })
  });
  auto pp = multivar_polynomial::PolynomialProduct<Poly, 2>(v.begin(), v.end());
  BOOST_TEST(Of(pp, Eigen::Vector2d::Zero()) == 4);
  BOOST_TEST(Of(pp, Eigen::Vector2d::Ones() * 2) == (1 + 4 + 12) * (4 + 10));
}


BOOST_AUTO_TEST_CASE(polynomial_product_D, * utf::tolerance(tt::fpc::percent_tolerance(1e-10)))
{
  using Poly = multivar_polynomial::Polynomial<double, int>;
  auto v = std::array<Poly, 2>({
    Poly({
      {0, 1},
      {1, 2},
      {2, 3}
    }),
    Poly({
      {0, 4},
      {1, 5}
    })
  });

  auto pp = multivar_polynomial::PolynomialProduct<Poly, 2>(v.begin(), v.end());
  auto dpp0 = multivar_polynomial::D(pp, 0);
  v = {
    Poly({
      {0, 2},
      {1, 6}
    }),
    Poly({
      {0, 4},
      {1, 5}
    })
  };
  BOOST_TEST(v.size() == dpp0.size());
  for (std::size_t i = 0; i != v.size(); ++i)
  {
    BOOST_TEST(v[i] == dpp0[i]);
  }
  auto dpp1 = multivar_polynomial::D(pp, 1);
  v = {
    Poly({
      {0, 1},
      {1, 2},
      {2, 3}
    }),
    Poly({
      {0, 5}
    })
  };
  BOOST_TEST(v.size() == dpp1.size());
  for (std::size_t i = 0; i != v.size(); ++i)
  {
    BOOST_TEST(v[i] == dpp1[i]);
  }
}

