#define BOOST_TEST_MODULE polynomial_product_unit_test

#include "boost/test/unit_test.hpp"
#include "mvPolynomial/polynomial_product.hpp"
#include "mvPolynomial/polynomial.hpp"

#include <vector>

namespace utf = boost::unit_test;
namespace tt  = boost::test_tools;

using Poly = mvPolynomial::Polynomial<int, double>;

BOOST_AUTO_TEST_CASE(polynomial_product_init, *utf::tolerance(tt::fpc::percent_tolerance(1e-10))) {
  auto v  = std::array<Poly, 2>({
      Poly({{0, 1}, {1, 2}, {2, 3}}
      ),
      Poly({{0, 4}, {1, 5}}
      )
  });
  auto pp = mvPolynomial::PolynomialProduct<Poly, 2>(v.begin(), v.end());
  BOOST_TEST(v.size() == pp.size());
  for (std::size_t i = 0; i != v.size(); ++i) {
    BOOST_TEST(v[i] == pp[i]);
  }
}

BOOST_AUTO_TEST_CASE(polynomial_product_Of, *utf::tolerance(tt::fpc::percent_tolerance(1e-10))) {
  auto v  = std::array<Poly, 2>({
      Poly({{0, 1}, {1, 2}, {2, 3}}
      ),
      Poly({{0, 4}, {1, 5}}
      )
  });
  auto pp = mvPolynomial::PolynomialProduct<Poly, 2>(v.begin(), v.end());
  BOOST_TEST(Of(pp, Eigen::Vector2d::Zero()) == 4);
  BOOST_TEST(Of(pp, Eigen::Vector2d::Ones() * 2) == (1 + 4 + 12) * (4 + 10));
}

BOOST_AUTO_TEST_CASE(polynomial_product_D, *utf::tolerance(tt::fpc::percent_tolerance(1e-10))) {
  auto v = std::array<Poly, 2>({
      Poly({{0, 1}, {1, 2}, {2, 3}}
      ),
      Poly({{0, 4}, {1, 5}}
      )
  });

  auto pp   = mvPolynomial::PolynomialProduct<Poly, 2>(v.begin(), v.end());
  auto dpp0 = mvPolynomial::D(pp, 0);
  v         = {
      Poly({{0, 2}, {1, 6}}
      ),
      Poly({{0, 4}, {1, 5}}
      )
  };
  BOOST_TEST(v.size() == dpp0.size());
  for (std::size_t i = 0; i != v.size(); ++i) {
    BOOST_TEST(v[i] == dpp0[i]);
  }
  auto dpp1 = mvPolynomial::D(pp, 1);
  v         = {
      Poly({{0, 1}, {1, 2}, {2, 3}}
      ),
      Poly({{0, 5}}
      )
  };
  BOOST_TEST(v.size() == dpp1.size());
  for (std::size_t i = 0; i != v.size(); ++i) {
    BOOST_TEST(v[i] == dpp1[i]);
  }
}

BOOST_AUTO_TEST_CASE(
    polynomial_product_Integrate, *utf::tolerance(tt::fpc::percent_tolerance(1e-10))
) {
  auto v = std::array<Poly, 2>({
      Poly({{0, 1}, {1, 2}, {2, 3}}
      ),
      Poly({{0, 4}, {1, 5}}
      )
  });

  auto pp   = mvPolynomial::PolynomialProduct<Poly, 2>(v.begin(), v.end());
  auto spp0 = mvPolynomial::Integrate(pp, 0);
  v         = {
      Poly({{1, 1}, {2, 1}, {3, 1}}
      ),
      Poly({{0, 4}, {1, 5}}
      )
  };
  BOOST_TEST(v.size() == spp0.size());
  for (std::size_t i = 0; i != v.size(); ++i) {
    BOOST_TEST(v[i] == spp0[i]);
  }
  auto spp1 = mvPolynomial::Integrate(pp, 1);
  v         = {
      Poly({{0, 1}, {1, 2}, {2, 3}}
      ),
      Poly({{1, 4}, {2, 2.5}}
      )
  };
  BOOST_TEST(v.size() == spp1.size());
  for (std::size_t i = 0; i != v.size(); ++i) {
    BOOST_TEST(v[i] == spp1[i]);
  }
}
