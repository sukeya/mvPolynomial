#ifndef _MVPOLYNOMIAL_MVPOLYNOMIAL_HPP_
#define _MVPOLYNOMIAL_MVPOLYNOMIAL_HPP_

#include "mvPolynomial/type.hpp"
#include "mvPolynomial/index_comparer.hpp"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <numeric>
#include <ranges>
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
  inline static R rel_tolerance = std::numeric_limits<R>::epsilon();
  inline static R abs_tolerance = std::numeric_limits<R>::min();

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

  MVPolynomial() : index2value_() { index2value_[index_type::Zero()] = R{0}; }

  explicit MVPolynomial(const allocator_type& allocator) : index2value_(allocator) {
    index2value_[index_type::Zero()] = R{0};
  }

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

  MVPolynomial(mapped_type r, const allocator_type& a = allocator_type{}) : index2value_(a) {
    index2value_[index_type::Zero()] = r;
  }

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

  size_type size() const noexcept { return index2value_.size(); }
  size_type max_size() const noexcept { return index2value_.max_size(); }
  size_type capacity() const noexcept { return index2value_.capacity(); }

  mapped_type& operator[](const key_type& index) {
    CheckIndexIncludingNegative(index);
    return index2value_[index];
  }
  mapped_type& operator[](key_type&& index) {
    CheckIndexIncludingNegative(index);
    return index2value_[index];
  }

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

  void DeleteZeroCoeffTerm() {
    std::vector<size_t> removed_term_indexes;
    for (size_t i = 0; const auto& index_and_coeff : index2value_) {
      if (std::abs(index_and_coeff.second) < abs_tolerance) {
        removed_term_indexes.push_back(i);
      }
      ++i;
    }
    for (auto removed_index : removed_term_indexes | std::views::reverse) {
      index2value_.erase(std::next(index2value_.begin(), removed_index));
    }
    if (index2value_.empty()) {
      index2value_[index_type::Zero()] = R{0.0};
    }
  }

  R operator()(const coord_type& x) const {
    std::vector<R> partial_sums;
    auto           partial_sum = rbegin()->second;
    for (auto it = rbegin(); it != std::prev(rend()); ++it) {
      const auto& index = it->first;

      auto        next_it    = std::next(it);
      auto        next_coeff = next_it->second;
      const auto& next_index = next_it->first;

      index_type index_diff = index - next_index;
      if ((index_diff > 0).all()) {
        partial_sum = next_coeff + partial_sum * (x.array().pow(index_diff.template cast<R>())).prod();
      } else {
        partial_sum *= (x.array().pow(index.template cast<R>())).prod();
        partial_sums.push_back(partial_sum);
        partial_sum = next_coeff;
      }
    }
    {
      const auto& first_index = begin()->first;
      partial_sum *= (x.array().pow(first_index.template cast<R>())).prod();
    }
    return std::reduce(partial_sums.cbegin(), partial_sums.cend()) + partial_sum;
  }

  MVPolynomial operator()(const MVPolynomial& mvp, int axis) const {
    details::CheckAxis(dim, axis);

    auto partial_sums = std::vector<MVPolynomial>{get_allocator()};
    auto partial_sum  = MVPolynomial{rbegin()->second, get_allocator()};
    for (auto it = rbegin(); it != std::prev(rend()); ++it) {
      const auto& index = it->first;

      auto        next_it    = std::next(it);
      auto        next_coeff = next_it->second;
      const auto& next_index = next_it->first;

      index_type index_diff              = index - next_index;
      index_type index_diff_without_axis = index_diff;
      index_diff_without_axis[axis]      = 0;

      if ((index_diff > 0).all()) {
        auto tmp_mvp = MVPolynomial{{{index_diff_without_axis, 1}}, get_allocator()};
        partial_sum  = next_coeff + partial_sum * (mvp.pow(index[axis])) * tmp_mvp;
      } else {
        index_type index_without_axis = index;
        index_without_axis[axis]      = 0;
        auto tmp_mvp                  = MVPolynomial{{{index_without_axis, 1}}, get_allocator()};

        partial_sum *= mvp.pow(index[axis]) * tmp_mvp;
        partial_sums.push_back(std::move(partial_sum));
        partial_sum = MVPolynomial{next_coeff, get_allocator()};
      }
    }
    {
      const auto& first_index              = begin()->first;
      index_type  first_index_without_axis = first_index;
      first_index_without_axis[axis]       = 0;
      auto tmp_mvp                         = MVPolynomial{{{first_index_without_axis, 1}}, get_allocator()};
      partial_sum *= mvp.pow(first_index[axis]) * tmp_mvp;
    }
    return std::reduce(partial_sums.cbegin(), partial_sums.cend()) + partial_sum;
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
    DeleteZeroCoeffTerm();
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
    DeleteZeroCoeffTerm();
    return *this;
  }

  MVPolynomial& operator-=(mapped_type r) {
    auto idx = index_type::Zero();
    if (contains(idx)) {
      (*this)[idx] -= r;
    } else {
      (*this)[idx] = -r;
    }
    DeleteZeroCoeffTerm();
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
    DeleteZeroCoeffTerm();
    return *this;
  }

  MVPolynomial& operator*=(mapped_type r) {
    for (auto& index_and_coeff : *this) {
      auto& coeff = index_and_coeff.second;
      coeff *= r;
    }
    DeleteZeroCoeffTerm();
    return *this;
  }

  MVPolynomial& operator*=(const MVPolynomial& r) {
    if (r.size() == 1) {
      const auto& [r_index, r_coeff] = *(r.begin());
      auto result                    = MVPolynomial(get_allocator());
      result.index2value_.clear();
      for (const auto& [index, coeff] : *this) {
        const auto new_index = index + r_index;
        const auto new_coeff = coeff * r_coeff;
        if (result.contains(new_index)) {
          result[new_index] += new_coeff;
        } else {
          result[new_index] = new_coeff;
        }
      }
      swap(result);
    } else {
      *this = *this * r;
    }
    DeleteZeroCoeffTerm();
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
      auto abs_diff = std::abs(l_coeff - r_coeff);
      if (abs_diff < 1) {
        if (!(abs_diff < abs_tolerance)) {
          return false;
        }
      } else {
        if (!(abs_diff < rel_tolerance * std::max(std::abs(l_coeff), std::abs(r_coeff)))) {
          return false;
        }
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

    mul.DeleteZeroCoeffTerm();

    return mul;
  }

 private:
  void CheckIndexIncludingNegative(const index_type& index) const {
    if ((index < 0).any()) {
      throw std::invalid_argument("Negative index not supported!");
    }
  }

  void CheckSelfIndexes() const {
    for (const auto& index_and_coeff : index2value_) {
      CheckIndexIncludingNegative(index_and_coeff.first);
    }
  }

  IndexContainer index2value_;
};

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
  dp.DeleteZeroCoeffTerm();

  return dp;
}

template <std::signed_integral IntType, std::floating_point R, int D, class Allocator>
auto Integrate(MVPolynomial<IntType, R, D, Allocator> p, int axis) {
  using MP = MVPolynomial<IntType, R, D, Allocator>;

  details::CheckAxis(D, axis);

  auto result = MP(std::initializer_list<typename MP::value_type>{}, p.get_allocator());
  for (const auto& [index, value] : p) {
    auto new_index = index;
    ++new_index[axis];
    const auto new_value = value / new_index[axis];
    if (result.contains(new_index)) {
      result[new_index] += new_value;
    } else {
      result[new_index] = new_value;
    }
  }

  result.DeleteZeroCoeffTerm();

  return result;
}

}  // namespace mvPolynomial

#endif
