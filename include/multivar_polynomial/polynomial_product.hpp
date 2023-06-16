#ifndef _MULTIVAR_POLYNOMIAL_POLYNOMIAL_PRODUCT_HPP_
#define _MULTIVAR_POLYNOMIAL_POLYNOMIAL_PRODUCT_HPP_


#include "multivar_polynomial/type.hpp"

#include <array>
#include <numeric>

#include "boost/math/tools/polynomial.hpp"
#include "boost/range/adaptor/indexed.hpp"
#include "Eigen/Core"


namespace multivar_polynomial
{
  template <class R, std::size_t Dim>
  class PolynomialProduct
  {
  public:
    inline static const std::size_t dim{Dim};

    using polynomial_type = boost::math::tools::polynomial<R>;
    using coord_type = Eigen::Vector<R, dim>;

  private:
    using PolynomialContainer = std::array<polynomial_type, dim>;
  
  public:
    using reference = PolynomialContainer::reference;
    using const_reference = PolynomialContainer::const_reference;

    using iterator = PolynomialContainer::iterator;
    using const_iterator = PolynomialContainer::const_iterator;
    using reverse_iterator = PolynomialContainer::reverse_iterator;
    using const_reverse_iterator = PolynomialContainer::const_reverse_iterator;

    using size_type = PolynomialContainer::size_type;
    using difference_type = PolynomialContainer::difference_type;

    using pointer = PolynomialContainer::pointer;
    using const_pointer = PolynomialContainer::const_pointer;

    using value_type = PolynomialContainer::value_type;


    PolynomialProduct() { this->fill(polynomial_type({1})); }

    PolynomialProduct(std::initializer_list<polynomial_type> l)
    {
      for(const auto& indexed_p : l | boost::adaptors::indexed())
      {
        auto i = indexed_p.index();
        const auto& p = indexed_p.value();
        polynomials_[i] = p;
      }
      // Initialize the rest polynomials.
      for(auto i = l.size(); i != dim; ++i)
      {
        polynomials_[i] = {1};
      }
    }

    PolynomialProduct(const PolynomialProduct&) = default;
    PolynomialProduct(PolynomialProduct&&) = default;
    PolynomialProduct& operator=(const PolynomialProduct&) = default;
    PolynomialProduct& operator=(PolynomialProduct&&) = default;
    virtual ~PolynomialProduct() = default;

    auto at(size_type n) { return polynomials_.at(n); }
    auto at(size_type n) const { return polynomials_.at(n); }

    auto operator[](size_type n) { return polynomials_[n]; }
    auto operator[](size_type n) const { return polynomials_[n]; }

    auto front() { return polynomials_.front(); }
    auto front() const { return polynomials_.front(); }

    auto back() { return polynomials_.back(); }
    auto back() const { return polynomials_.back(); }

    iterator begin() noexcept { return polynomials_.begin(); }
    const_iterator begin() const noexcept  { return polynomials_.begin(); }

    iterator end() noexcept { return polynomials_.end(); }
    const_iterator end() const noexcept { return polynomials_.end(); }

    reverse_iterator rbegin() noexcept { return polynomials_.rbegin(); }
    const_reverse_iterator rbegin() const noexcept { return polynomials_.rbegin(); }

    reverse_iterator rend() noexcept { return polynomials_.rend(); }
    const_reverse_iterator rend() const noexcept { return polynomials_.rend(); }
    
    const_iterator cbegin() const noexcept { return polynomials_.cbegin(); }
    const_iterator cend() const noexcept { return polynomials_.rend(); }

    const_reverse_iterator crbegin() const noexcept { return polynomials_.crbegin(); }
    const_reverse_iterator crend() const noexcept { return polynomials_.crbegin(); }

    bool empty() const noexcept { return polynomials_.empty(); }
    size_type size() const noexcept { return polynomials_.size(); }
    size_type max_size() const noexcept { return polynomials_.max_size(); }

    void fill(const polynomial_type& p) { polynomials_.fill(p); }
    void swap(PolynomialContainer& other) { polynomials_.swap(other); }

  private:
    std::array<polynomial_type, Dim> polynomials_;
  };


  template <class T, class R, int Dim>
  auto OfPPImpl(const T& indexed_p, const CoordType<R, Dim>& x)
  {
    auto index = indexed_p.index();
    auto p = indexed_p.value();
    return p[x[index]];
  }


  template <class R, std::size_t Dim>
  auto Of(const PolynomialProduct<R, Dim>& pp, const typename PolynomialProduct<R, Dim>::coord_type& x)
  {
    auto indexed_pp = boost::adaptors::index(pp);
    return std::reduce(
      boost::begin(indexed_pp),
      boost::end(indexed_pp),
      [&x](const auto& p, const auto& q){ return OfPPImpl(p, x) * OfPPImpl(q, x);}
    );
  }


  template <class R, std::size_t Dim>
  auto D(const PolynomialProduct<R, Dim>& p, std::size_t axis)
  {
    auto q = PolynomialProduct<R, Dim>(p);
    q[axis] = p.at(axis).prime();
    return q;
  }


  template <class R, std::size_t Dim>
  auto Integral(const PolynomialProduct<R, Dim>& p, std::size_t axis)
  {
    auto q = PolynomialProduct<R, Dim>(p);
    q[axis] = p.at(axis).integrate();
    return q;
  }
}

#endif
