#ifndef _MULTIVAR_POLYNOMIAL_POLYNOMIAL_PRODUCT_HPP_
#define _MULTIVAR_POLYNOMIAL_POLYNOMIAL_PRODUCT_HPP_

#include "multivar_polynomial/type.hpp"

#include <array>
#include <numeric>

#include "boost/range/adaptor/indexed.hpp"
#include "boost/tuple/tuple.hpp"
#include "boost/iterator/zip_iterator.hpp"
#include "Eigen/Core"
#include "fmt/core.h"

namespace multivar_polynomial {
template <class P, class R>
P MakeScalarPolynomial(R r) {
  return P(r);
}

template <class P, std::size_t Dim>
class PolynomialProduct {
 public:
  inline static const std::size_t dim{Dim};

  using polynomial_type = P;
  using coord_type      = CoordType<typename polynomial_type::coord_type, dim>;

 private:
  using PolynomialContainer = std::array<polynomial_type, dim>;

 public:
  using reference       = PolynomialContainer::reference;
  using const_reference = PolynomialContainer::const_reference;

  using iterator               = PolynomialContainer::iterator;
  using const_iterator         = PolynomialContainer::const_iterator;
  using reverse_iterator       = PolynomialContainer::reverse_iterator;
  using const_reverse_iterator = PolynomialContainer::const_reverse_iterator;

  using size_type       = PolynomialContainer::size_type;
  using difference_type = PolynomialContainer::difference_type;

  using pointer       = PolynomialContainer::pointer;
  using const_pointer = PolynomialContainer::const_pointer;

  using value_type = PolynomialContainer::value_type;

  PolynomialProduct() { this->fill(MakeScalarPolynomial<polynomial_type>(1)); }

  template <typename InputIterator>
  PolynomialProduct(InputIterator s, InputIterator e) {
    if (std::distance(s, e) != polynomials_.size()) {
      throw std::runtime_error(
          fmt::format(
              "PolynomialProduct: The size of given initializer list must be equal to the "
              "dimension {}",
              dim
          )
      );
    }
    std::for_each(
        boost::make_zip_iterator(boost::make_tuple(s, polynomials_.begin())),
        boost::make_zip_iterator(boost::make_tuple(e, polynomials_.end())),
        [](const auto& t) { boost::tuples::get<1>(t) = boost::tuples::get<0>(t); }
    );
  }

  explicit PolynomialProduct(std::initializer_list<polynomial_type> l) {
    if (l.size() != polynomials_.size()) {
      throw std::runtime_error(
          fmt::format(
              "PolynomialProduct: The size of given initializer list must be equal to the "
              "dimension {}",
              dim
          )
      );
    }

    for (const auto& indexed_p : l | boost::adaptors::indexed()) {
      auto        i   = indexed_p.index();
      const auto& p   = indexed_p.value();
      polynomials_[i] = p;
    }
  }

  PolynomialProduct(const PolynomialProduct&)            = default;
  PolynomialProduct(PolynomialProduct&&)                 = default;
  PolynomialProduct& operator=(const PolynomialProduct&) = default;
  PolynomialProduct& operator=(PolynomialProduct&&)      = default;
  virtual ~PolynomialProduct()                           = default;

  value_type&       at(size_type n) { return polynomials_.at(n); }
  const value_type& at(size_type n) const { return polynomials_.at(n); }

  value_type&       operator[](size_type n) { return polynomials_[n]; }
  const value_type& operator[](size_type n) const { return polynomials_[n]; }

  reference       front() { return polynomials_.front(); }
  const_reference front() const { return polynomials_.front(); }

  reference       back() { return polynomials_.back(); }
  const_reference back() const { return polynomials_.back(); }

  iterator       begin() noexcept { return polynomials_.begin(); }
  const_iterator begin() const noexcept { return polynomials_.begin(); }

  iterator       end() noexcept { return polynomials_.end(); }
  const_iterator end() const noexcept { return polynomials_.end(); }

  reverse_iterator       rbegin() noexcept { return polynomials_.rbegin(); }
  const_reverse_iterator rbegin() const noexcept { return polynomials_.rbegin(); }

  reverse_iterator       rend() noexcept { return polynomials_.rend(); }
  const_reverse_iterator rend() const noexcept { return polynomials_.rend(); }

  const_iterator cbegin() const noexcept { return polynomials_.cbegin(); }
  const_iterator cend() const noexcept { return polynomials_.cend(); }

  const_reverse_iterator crbegin() const noexcept { return polynomials_.crbegin(); }
  const_reverse_iterator crend() const noexcept { return polynomials_.crend(); }

  bool      empty() const noexcept { return polynomials_.empty(); }
  size_type size() const noexcept { return polynomials_.size(); }
  size_type max_size() const noexcept { return polynomials_.max_size(); }

  void fill(const polynomial_type& p) { polynomials_.fill(p); }
  void swap(PolynomialProduct& other) { polynomials_.swap(other.polynomials_); }

  // TODO consider tolerance.
  friend bool operator==(const PolynomialProduct& l, const PolynomialProduct& r) {
    for (std::size_t i = 0; i != dim; ++i) {
      if (l[i] != r[i]) {
        return false;
      }
    }
    return true;
  }

  // TODO consider tolerance.
  friend bool operator!=(const PolynomialProduct& l, const PolynomialProduct& r) {
    return !(l == r);
  }

  friend PolynomialProduct operator*(const PolynomialProduct& l, const PolynomialProduct& r) {
    auto mul = PolynomialProduct();
    for (std::size_t i = 0; i != dim; ++i) {
      mul[i] = l[i] * r[i];
    }
    return mul;
  }

 private:
  PolynomialContainer polynomials_;
};

template <class P, std::size_t Dim>
auto Of(
    const PolynomialProduct<P, Dim>& pp, const typename PolynomialProduct<P, Dim>::coord_type& x
) {
  auto mul = typename PolynomialProduct<P, Dim>::polynomial_type::mapped_type(1);
  for (const auto& index_and_pp : pp | boost::adaptors::indexed()) {
    auto        index = index_and_pp.index();
    const auto& p     = index_and_pp.value();
    mul *= Of(p, x[index]);
  }
  return mul;
}

template <class P, std::size_t Dim>
auto D(const PolynomialProduct<P, Dim>& p, std::size_t axis) {
  auto q     = PolynomialProduct<P, Dim>(p);
  q.at(axis) = D(p.at(axis));
  return q;
}

template <class P, std::size_t Dim>
auto Integrate(const PolynomialProduct<P, Dim>& p, std::size_t axis) {
  auto q     = PolynomialProduct<P, Dim>(p);
  q.at(axis) = Integrate(p.at(axis));
  return q;
}
}  // namespace multivar_polynomial

#endif
