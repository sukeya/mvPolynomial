#ifndef _MVPOLYNOMIAL_MVPOLYNOMIAL_HPP_
#define _MVPOLYNOMIAL_MVPOLYNOMIAL_HPP_

#include "mvPolynomial/type.hpp"
#include "mvPolynomial/index_comparer.hpp"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "fmt/core.h"
#include "platanus/btree_map.hpp"

namespace mvPolynomial {
namespace details {
inline void CheckAxis(int dim, int axis) {
  if (axis < 0 || axis >= dim) {
    throw std::runtime_error(fmt::format("CheckAxis: Given axis {} must be in [0, {}).", axis, dim));
  }
}
}  // namespace details

template <
    std::signed_integral IntType,
    std::floating_point  R,
    int                  D,
    class Allocator = std::allocator<std::pair<const IndexType<IntType, D>, R>>>
class MVPolynomial final {
 public:
  static_assert(D > 0, "MVPolynomial: the dimension must be greater than 0.");

  static constexpr int dim = D;

  // This setting is too strict, so I expect users to set tolerance.
  static R tolerance;

  using index_type = IndexType<IntType, D>;
  using coord_type = CoordType<R, dim>;

 private:
  using IndexContainer = platanus::btree_map<index_type, R, IndexComparer<IntType, D>, Allocator>;

 public:
  using key_type    = IndexContainer::key_type;
  using value_type  = IndexContainer::value_type;
  using mapped_type = IndexContainer::mapped_type;
  using coeff_type  = mapped_type;

  using key_compare   = IndexContainer::key_compare;
  using value_compare = IndexContainer::value_compare;

  using allocator_type = IndexContainer::allocator_type;

  using pointer       = IndexContainer::pointer;
  using const_pointer = IndexContainer::const_pointer;

  using reference       = IndexContainer::reference;
  using const_reference = IndexContainer::const_reference;

  using size_type       = IndexContainer::size_type;
  using difference_type = IndexContainer::difference_type;

  using iterator               = IndexContainer::iterator;
  using const_iterator         = IndexContainer::const_iterator;
  using reverse_iterator       = IndexContainer::reverse_iterator;
  using const_reverse_iterator = IndexContainer::const_reverse_iterator;

  MVPolynomial(const MVPolynomial& other)            = default;
  MVPolynomial& operator=(const MVPolynomial& other) = default;
  MVPolynomial(MVPolynomial&& other)                 = default;
  MVPolynomial& operator=(MVPolynomial&& other)      = default;
  ~MVPolynomial()                                    = default;

  MVPolynomial()
      : index2value_({
            {index_type::Zero(), 0}
  }) {}

  explicit MVPolynomial(const allocator_type& allocator)
      : index2value_(
            {
                {index_type::Zero(), 0}
  },
            allocator
        ) {}

  template <typename InputIterator>
  MVPolynomial(InputIterator s, InputIterator e) : index2value_(s, e) {
    CheckSelfIndexes();
  }

  template <typename InputIterator>
  MVPolynomial(InputIterator s, InputIterator e, const allocator_type& allocator) : index2value_(s, e, allocator) {
    CheckSelfIndexes();
  }

  MVPolynomial(std::initializer_list<value_type> l, const allocator_type& a = allocator_type{}) : index2value_(l, a) {
    CheckSelfIndexes();
  }

  MVPolynomial(const MVPolynomial& m, const allocator_type& a) : index2value_(m.index2value_, a) {}

  MVPolynomial(MVPolynomial&& m, const allocator_type& a) : index2value_(std::move(m.index2value_), a) {}

  MVPolynomial& operator=(std::initializer_list<value_type> l) {
    index2value_.clear();
    index2value_.insert(l.begin(), l.end());
    CheckSelfIndexes();
    return *this;
  }

  MVPolynomial(mapped_type r, const allocator_type& a = allocator_type{}) : index2value_({{index_type::Zero(), r}}, a) {}

  allocator_type get_allocator() const noexcept { return index2value_.get_allocator(); }
  key_compare    key_comp() const noexcept { return index2value_.key_comp(); }

  iterator       begin() noexcept { return index2value_.begin(); }
  const_iterator begin() const noexcept { return index2value_.begin(); }

  iterator       end() noexcept { return index2value_.end(); }
  const_iterator end() const noexcept { return index2value_.end(); }

  reverse_iterator       rbegin() noexcept { return index2value_.rbegin(); }
  const_reverse_iterator rbegin() const noexcept { return index2value_.rbegin(); }

  reverse_iterator       rend() noexcept { return index2value_.rend(); }
  const_reverse_iterator rend() const noexcept { return index2value_.rend(); }

  const_iterator cbegin() const noexcept { return index2value_.cbegin(); }
  const_iterator cend() const noexcept { return index2value_.cend(); }

  const_reverse_iterator crbegin() const noexcept { return index2value_.crbegin(); }
  const_reverse_iterator crend() const noexcept { return index2value_.crend(); }

  bool empty() const noexcept { return index2value_.empty(); }

  size_type size() const noexcept { return index2value_.size(); }
  size_type max_size() const noexcept { return index2value_.max_size(); }
  size_type capacity() const noexcept { return index2value_.capacity(); }

  mapped_type& operator[](const key_type& index) { return index2value_[index]; }
  mapped_type& operator[](key_type&& index) { return index2value_[index]; }

  mapped_type&       at(const key_type& i) { return index2value_.at(i); }
  const mapped_type& at(const key_type& i) const { return index2value_.at(i); }

  void swap(MVPolynomial& m) { index2value_.swap(m.index2value_); }

  iterator       find(const key_type& i) { return index2value_.find(i); }
  const_iterator find(const key_type& i) const { return index2value_.find(i); }
  template <typename K>
  iterator find(const K& i) {
    return index2value_.find(i);
  }
  template <typename K>
  const_iterator find(const K& i) const {
    return index2value_.find(i);
  }

  bool contains(const key_type& i) const { return index2value_.contains(i); }
  template <typename K>
  bool contains(const K& i) const {
    return index2value_.contains(i);
  }

  iterator       lower_bound(const key_type& i) { return index2value_.lower_bound(i); }
  const_iterator lower_bound(const key_type& i) const { return index2value_.lower_bound(i); }
  template <typename K>
  iterator lower_bound(const K& i) {
    return index2value_.lower_bound(i);
  }
  template <typename K>
  const_iterator lower_bound(const K& i) const {
    return index2value_.lower_bound(i);
  }

  iterator       upper_bound(const key_type& i) { return index2value_.upper_bound(i); }
  const_iterator upper_bound(const key_type& i) const { return index2value_.upper_bound(i); }
  template <typename K>
  iterator upper_bound(const K& i) {
    return index2value_.upper_bound(i);
  }
  template <typename K>
  const_iterator upper_bound(const K& i) const {
    return index2value_.upper_bound(i);
  }

  MVPolynomial pow(int exp) const {
    if (exp < 0) {
      throw std::invalid_argument("Given exp must be positive.");
    }
    switch (exp) {
      case 0:
        return MVPolynomial{1, get_allocator()};
      case 1:
        return *this;
      case 2:
        return (*this) * (*this);
      default:
        auto max_pow2_under_exp = std::bit_floor(static_cast<unsigned int>(exp));
        auto max_bit_width      = std::bit_width(max_pow2_under_exp);
        auto cache              = std::vector<MVPolynomial>(max_bit_width - 1, get_allocator());
        cache.at(0)             = (*this) * (*this);
        for (int i = 2; i < max_bit_width; ++i) {
          cache.at(i - 1) = cache.at(i - 2) * cache.at(i - 2);
        }

        assert(!cache.empty());
        auto powed_mvp = std::move(cache.back());
        exp -= max_pow2_under_exp;
        while (exp > 1) {
          auto max_pow2_under_exp = std::bit_floor(static_cast<unsigned int>(exp));
          auto max_bit_width      = std::bit_width(max_pow2_under_exp);
          powed_mvp *= cache[max_bit_width - 2];
          exp -= max_pow2_under_exp;
        }
        if (exp == 1) {
          powed_mvp *= *this;
        }
        return powed_mvp;
    }
  }

  R operator()(const coord_type& x) const {
    auto sum = R(0.0);
    for (const auto& index_and_coeff : index2value_) {
      auto        coeff = index_and_coeff.second;
      const auto& index = index_and_coeff.first;
      sum += coeff * (x.array().pow(index.template cast<double>())).prod();
    }
    return sum;
  }

  MVPolynomial operator()(const MVPolynomial& mvp, int axis) const {
    details::CheckAxis(dim, axis);

    auto composed_mvp = MVPolynomial{get_allocator()};
    for (const auto& index_and_coeff : index2value_) {
      auto        coeff = index_and_coeff.second;
      const auto& index = index_and_coeff.first;

      auto index_with_axis_0  = index;
      index_with_axis_0[axis] = 0;
      composed_mvp += MVPolynomial{{{index_with_axis_0, coeff}}, get_allocator()} * mvp.pow(index[axis]);
    }
    return composed_mvp;
  }

  MVPolynomial operator+() const { return *this; }

  MVPolynomial operator-() const& {
    auto m = MVPolynomial(*this, get_allocator());
    for (auto& i_and_v : m) {
      auto& [_, v] = i_and_v;
      v            = -v;
    }
    return m;
  }

  MVPolynomial operator-() && {
    for (auto& i_and_v : *this) {
      auto& [_, v] = i_and_v;
      v            = -v;
    }
    return std::move(*this);
  }

  MVPolynomial& operator+=(mapped_type r) {
    auto idx = index_type::Zero();
    if (contains(idx)) {
      (*this)[idx] += r;
    } else {
      (*this)[idx] = r;
    }
    return *this;
  }

  MVPolynomial& operator+=(const MVPolynomial& r) {
    for (const auto& [idx, coeff] : r) {
      if (contains(idx)) {
        (*this)[idx] += coeff;
      } else {
        (*this)[idx] = coeff;
      }
    }
    return *this;
  }

  MVPolynomial& operator-=(mapped_type r) {
    auto idx = index_type::Zero();
    if (contains(idx)) {
      (*this)[idx] -= r;
    } else {
      (*this)[idx] = -r;
    }
    return *this;
  }

  MVPolynomial& operator-=(const MVPolynomial& r) {
    for (const auto& [idx, coeff] : r) {
      if (contains(idx)) {
        (*this)[idx] -= coeff;
      } else {
        (*this)[idx] = -coeff;
      }
    }
    return *this;
  }

  MVPolynomial& operator*=(mapped_type r) {
    auto idx = index_type::Zero();
    for (auto& index_and_coeff : *this) {
      auto& coeff = index_and_coeff.second;
      coeff *= r;
    }
    return *this;
  }

  MVPolynomial& operator*=(const MVPolynomial& r) {
    if (r.size() == 1) {
      const auto& [r_index, r_coeff] = *(r.begin());
      for (auto& index_and_coeff : *this) {
        auto& index = const_cast<index_type&>(index_and_coeff.first);
        auto& coeff = index_and_coeff.second;
        index += r_index;
        coeff *= r_coeff;
      }
    } else {
      *this = *this * r;
    }
    return *this;
  }

  // friend functions
  friend bool operator==(const MVPolynomial& l, const MVPolynomial& r) {
    using size_type = typename MVPolynomial::size_type;

    if (l.size() != r.size()) {
      return false;
    }
    auto l_it = l.cbegin();
    auto r_it = r.cbegin();
    for (size_type i = 0; i != l.size(); ++i) {
      const auto& [l_idx, l_coeff] = *l_it;
      const auto& [r_idx, r_coeff] = *r_it;
      if ((l_idx != r_idx).any()) {
        return false;
      }
      if (std::abs(l_coeff - r_coeff) >= tolerance) {
        return false;
      }
      ++l_it;
      ++r_it;
    }
    return true;
  }

  friend bool operator!=(const MVPolynomial& l, const MVPolynomial& r) { return !(l == r); }

  friend MVPolynomial operator+(const MVPolynomial& l, const MVPolynomial& r) {
    return MVPolynomial(l, l.get_allocator()) + r;
  }

  friend MVPolynomial operator+(MVPolynomial&& l, const MVPolynomial& r) {
    l += r;
    return std::move(l);
  }

  friend MVPolynomial operator+(const MVPolynomial& l, MVPolynomial&& r) { return std::move(r) + l; }

  friend MVPolynomial operator+(MVPolynomial&& l, MVPolynomial&& r) { return std::move(l) + r; }

  friend MVPolynomial operator-(const MVPolynomial& l, const MVPolynomial& r) {
    return MVPolynomial(l, l.get_allocator()) - r;
  }

  friend MVPolynomial operator-(MVPolynomial&& l, const MVPolynomial& r) {
    l -= r;
    return std::move(l);
  }

  friend MVPolynomial operator-(const MVPolynomial& l, MVPolynomial&& r) { return -std::move(r) + l; }

  friend MVPolynomial operator-(MVPolynomial&& l, MVPolynomial&& r) { return std::move(l) - r; }

  friend MVPolynomial operator*(const MVPolynomial& l, const MVPolynomial& r) {
    auto comparer = l.key_comp();

    auto mul = MVPolynomial(l.get_allocator());
    mul.index2value_.clear();
    // Calculate all product of each l's term and r's term.
    for (const auto& l_p : l) {
      const auto& [l_idx, l_v] = l_p;
      for (const auto& r_p : r) {
        const auto& [r_idx, r_v] = r_p;
        const auto idx           = l_idx + r_idx;
        const auto v             = l_v * r_v;
        if (mul.contains(idx)) {
          mul[idx] += v;
        } else {
          mul[idx] = v;
        }
      }
    }

    return mul;
  }

 private:
  void CheckSelfIndexes() const {
    for (const auto& index_and_coeff : index2value_) {
      if ((index_and_coeff.first < 0).any()) {
        throw std::invalid_argument("Negative index not supported!");
      }
    }
  }

  IndexContainer index2value_;
};

template <std::signed_integral IntType, std::floating_point R, int Dim, class Allocator>
R MVPolynomial<IntType, R, Dim, Allocator>::tolerance =
    std::ldexp(std::numeric_limits<double>::epsilon(), std::numeric_limits<double>::min_exponent);

template <std::signed_integral IntType, std::floating_point R, int Dim, class Allocator>
auto D(const MVPolynomial<IntType, R, Dim, Allocator>& p, int axis) {
  using MP            = MVPolynomial<IntType, R, Dim, Allocator>;
  using Index         = typename MP::index_type;
  using IndexAndCoeff = typename MP::value_type;

  details::CheckAxis(MP::dim, axis);

  auto dp   = MP{p.get_allocator()};
  auto p_it = p.begin();
  while (p_it != p.end()) {
    auto value = p_it->second;
    auto index = Index{p_it->first};
    if (index[axis] == 0) {
      auto d_end_it = p.end();
      for (int ith_axis = 0; ith_axis <= axis; ++ith_axis) {
        d_end_it = std::partition_point(p_it, d_end_it, [ith_axis, &index](const IndexAndCoeff& v) {
          return v.first[ith_axis] == index[ith_axis];
        });
      }
      // Skip indexes which axis-th element is zero.
      p_it = d_end_it;
    } else {
      value *= index[axis]--;
      dp[index] = value;
      ++p_it;
    }
  }
  return dp;
}

template <std::signed_integral IntType, std::floating_point R, int D, class Allocator>
auto Integrate(MVPolynomial<IntType, R, D, Allocator> p, int axis) {
  using MP = MVPolynomial<IntType, R, D, Allocator>;

  details::CheckAxis(D, axis);

  for (auto& index_and_value : p) {
    auto& value = index_and_value.second;
    auto& index = const_cast<typename MP::index_type&>(index_and_value.first);
    value /= ++index[axis];
  }
  return std::move(p);
}

}  // namespace mvPolynomial

#endif
