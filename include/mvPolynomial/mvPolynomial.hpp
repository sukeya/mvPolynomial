#ifndef _MVPOLYNOMIAL_MVPOLYNOMIAL_HPP_
#define _MVPOLYNOMIAL_MVPOLYNOMIAL_HPP_

#include "mvPolynomial/type.hpp"
#include "mvPolynomial/index_comparer.hpp"
#include "mvPolynomial/expression.hpp"

#include <algorithm>
#include <iterator>
#include <type_traits>

#include "Eigen/Core"
#include "fmt/core.h"
#include "platanus/btree_map.hpp"

namespace mvPolynomial {
namespace details {
inline void CheckAxis(int dim, int axis) {
  if (axis < 0 || axis >= dim) {
    throw std::runtime_error(
        fmt::format("CheckAxis: Given axis {} must be in [0, {}).", axis, dim)
    );
  }
}

template <class Iterator, class Coord>
auto OfImpl(Iterator begin, Iterator end, int dim, int axis, const Coord& x) {
  using Index = std::remove_cvref_t<typename Iterator::value_type::first_type>;

  CheckAxis(dim, axis);

  if (axis == dim - 1) {
    auto last_coeff = begin->second;
    auto last_index = Index{begin->first};
    for (auto it = std::next(begin); it != end; ++it) {
      const auto& [next_index, next_coeff] = *it;
      last_coeff *= std::pow(x[axis], last_index[axis] - next_index[axis]);
      last_coeff += next_coeff;
      last_index = next_index;
    }
    last_coeff *= x.pow(last_index.template cast<typename Coord::Scalar>()).prod();
    return last_coeff;
  } else {
    auto sum = typename Iterator::value_type::second_type(0);
    while (true) {
      const auto& [first_index, first_coeff] = *begin;
      auto partition_point                   = std::partition_point(
          begin,
          end,
          [axis, &first_index](const typename Iterator::value_type& pair) {
            return pair.first[axis] == first_index[axis];
          }
      );
      sum += OfImpl(begin, partition_point, dim, axis + 1, x);
      if (partition_point == end) {
        // The calculation ends.
        break;
      }
      begin = partition_point;
    }
    return sum;
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
  using Comparer       = IndexComparer<IntType, D>;
  using IndexContainer = platanus::btree_map<index_type, R, Comparer, Allocator>;

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
  MVPolynomial(InputIterator s, InputIterator e, const allocator_type& allocator)
      : index2value_(s, e, allocator) {
    CheckSelfIndexes();
  }

  MVPolynomial(std::initializer_list<value_type> l, const allocator_type& a = allocator_type{})
      : index2value_(l, a) {
    CheckSelfIndexes();
  }

  MVPolynomial(const MVPolynomial& m, const allocator_type& a) : index2value_(m.index2value_, a) {}

  MVPolynomial(MVPolynomial&& m, const allocator_type& a)
      : index2value_(std::move(m.index2value_), a) {}

  MVPolynomial& operator=(std::initializer_list<value_type> l) {
    index2value_.insert(l.begin(), l.end());
    CheckSelfIndexes();
    return *this;
  }

  MVPolynomial(mapped_type r) { index2value_.at(index_type::Zero()) = r; }

  allocator_type get_allocator() const noexcept { return index2value_.get_allocator(); }

  Comparer key_comp() const noexcept { return index2value_.key_comp(); }

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

  bool      empty() const noexcept { return index2value_.empty(); }
  size_type size() const noexcept { return index2value_.size(); }
  size_type max_size() const noexcept { return index2value_.max_size(); }
  size_type capacity() const noexcept { return index2value_.capacity(); }

  mapped_type& operator[](const key_type& index) { return index2value_[index]; }
  mapped_type& operator[](key_type&& index) { return index2value_[index]; }

  const mapped_type& operator[](const key_type& index) const {
    return const_cast<MVPolynomial*>(this)->operator[](index);
  }
  const mapped_type& operator[](key_type&& index) const {
    return const_cast<MVPolynomial*>(this)->operator[](index);
  }

  mapped_type&       at(const key_type& i) { return index2value_.at(i); }
  const mapped_type& at(const key_type& i) const { return index2value_.at(i); }

  void swap(MVPolynomial& m) { index2value_.swap(m.index2value_); }

  // I didn't want to add insert and erase, but for efficiency, I did.
  std::pair<iterator, bool> insert(const value_type& x) { return index2value_.insert(x); }
  std::pair<iterator, bool> insert(value_type&& x) { return index2value_.insert(std::move(x)); }

  iterator insert(iterator hint, const value_type& x) { return index2value_.insert(hint, x); }
  iterator insert(iterator hint, value_type&& x) { return index2value_.insert(hint, std::move(x)); }

  template <typename InputIterator>
  void insert(InputIterator b, InputIterator e) {
    index2value_.insert(b, e);
  }

  void insert(std::initializer_list<value_type> list) {
    index2value_.insert(list.begin(), list.end());
  }

  size_type erase(const key_type& key) { return index2value_.erase(key); }
  iterator  erase(const iterator& iter) { return index2value_.erase(iter); }
  size_type erase(const iterator& b, const iterator& e) { return index2value_.erase(b, e); }

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

  R operator()(const coord_type& x) const { return details::OfImpl(crbegin(), crend(), dim, 0, x); }

  MVPolynomial operator+() const { return *this; }

  MVPolynomial operator-() const& {
    auto m = MVPolynomial(*this);
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
    return MVPolynomial(l) + r;
  }

  friend MVPolynomial operator+(MVPolynomial&& l, const MVPolynomial& r) {
    l += r;
    return std::move(l);
  }

  friend MVPolynomial operator+(const MVPolynomial& l, MVPolynomial&& r) {
    return std::move(r) + l;
  }

  friend MVPolynomial operator+(MVPolynomial&& l, MVPolynomial&& r) { return std::move(l) + r; }

  friend MVPolynomial operator-(const MVPolynomial& l, const MVPolynomial& r) {
    return MVPolynomial(l) - r;
  }

  friend MVPolynomial operator-(MVPolynomial&& l, const MVPolynomial& r) {
    l -= r;
    return std::move(l);
  }

  friend MVPolynomial operator-(const MVPolynomial& l, MVPolynomial&& r) {
    return -std::move(r) + l;
  }

  friend MVPolynomial operator-(MVPolynomial&& l, MVPolynomial&& r) { return std::move(l) - r; }

  friend MVPolynomial operator*(const MVPolynomial& l, const MVPolynomial& r) {
    auto comparer = l.key_comp();

    if (l.size() == 1) {
      auto mul = r;
      mul *= l;
      return mul;
    }
    if (r.size() == 1) {
      return r * l;
    }

    auto mul = MVPolynomial(l.get_allocator());
    // Clear
    mul.erase(mul.begin());
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
    // The first index is the lowest index of all index,
    // so I only have to check if each of its elements is non-negative.
    if ((index2value_.begin()->first < 0).any()) {
      throw std::invalid_argument(fmt::format("Negative index not supported!"));
    }
  }

  IndexContainer index2value_;
};

template <std::signed_integral IntType, std::floating_point R, int Dim, class Allocator>
R MVPolynomial<IntType, R, Dim, Allocator>::tolerance = std::numeric_limits<R>::min_exponent10;

template <std::signed_integral IntType, std::floating_point R, int Dim, class Allocator>
auto D(int axis, const MVPolynomial<IntType, R, Dim, Allocator>& p) {
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
auto Integrate(MVPolynomial<IntType, R, D, Allocator>&& p, int axis) {
  using MP = MVPolynomial<IntType, R, D, Allocator>;

  details::CheckAxis(D, axis);

  for (auto& index_and_value : p) {
    auto& value = index_and_value.second;
    auto& index = const_cast<typename MP::index_type&>(index_and_value.first);
    value /= ++index[axis];
  }
  return std::move(p);
}

template <std::signed_integral IntType, std::floating_point R, int D, class Allocator>
auto Integrate(const MVPolynomial<IntType, R, D, Allocator>& p, int axis) {
  return Integrate(MVPolynomial<IntType, R, D, Allocator>(p), axis);
}

}  // namespace mvPolynomial

#endif
