#ifndef _MVPOLYNOMIAL_MVPOLYNOMIAL_HPP_
#define _MVPOLYNOMIAL_MVPOLYNOMIAL_HPP_

#include "mvPolynomial/type.hpp"
#include "mvPolynomial/index_comparer.hpp"
#include "mvPolynomial/polynomial.hpp"

#include <algorithm>
#include <iterator>
#include <memory>
#include <sstream>
#include <ranges>

#include "boost/tuple/tuple.hpp"
#include "boost/iterator/zip_iterator.hpp"
#include "Eigen/Core"
#include "fmt/core.h"

namespace mvPolynomial {
namespace {
void CheckAxis(int dim, int axis) {
  if (axis < 0 || axis >= dim) {
    throw std::runtime_error(
        fmt::format("CheckAxis: Given axis {} must be in [0, {}).", axis, dim)
    );
  }
}
}  // namespace

template <
    std::signed_integral IntType,
    std::floating_point  R,
    int                  D,
    class Allocator = std::allocator<std::pair<IndexType<IntType, D>, R>>>
class MVPolynomial final {
 public:
  static_assert(D > 0, "MVPolynomial: the dimension must be greater than 0.");

  static constexpr int dim = D;

  // This setting is too strict, so I expect users to set tolerance.
  static R tolerance = std::numeric_limits<R>::min_exponent10();

  using index_type = IndexType<IntType, D>;
  using coord_type = CoordType<R, dim>;

 private:
  using Compare        = IndexComparer<IntType, D>;
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

  MVPolynomial()                                     = default;
  MVPolynomial(const MVPolynomial& other)            = default;
  MVPolynomial& operator=(const MVPolynomial& other) = default;
  MVPolynomial(MVPolynomial&& other)                 = default;
  MVPolynomial& operator=(MVPolynomial&& other)      = default;
  ~MVPolynomial()                                    = default;

  explicit MVPolynomial(const allocator_type& allocator) : index2value_(allocator) {
    CheckSelfIndexes();
  }

  template <typename InputIterator>
  MVPolynomial(InputIterator s, InputIterator e) : index2value_(s, e) {
    CheckSelfIndexes();
  }

  template <typename InputIterator>
  explicit MVPolynomial(InputIterator s, InputIterator e, const allocator_type& allocator)
      : index2value_(s, e, allocator) {
    CheckSelfIndexes();
  }

  MVPolynomial(std::initializer_list<value_type> l, const allocator_type& a = allocator_type{})
      : index2value_(l, a) {
    CheckSelfIndexes();
  }

  explicit MVPolynomial(const MVPolynomial& m, const allocator_type& a)
      : index2value_(m.index2value_, a) {}

  explicit MVPolynomial(MVPolynomial&& m, const allocator_type& a)
      : index2value_(std::move(m.index2value_), a) {}

  MVPolynomial& operator=(std::initializer_list<value_type> l) {
    index2value_.insert(l.begin(), l.end());
    CheckSelfIndexes();
    return *this;
  }

  explicit MVPolynomial(mapped_type r) { index2value_.at(index_type::Zero()) = r; }

  allocator_type get_allocator() const noexcept { return index2value_.get_allocator(); }

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

  const mapped_type& operator[](const key_type& index) const {
    return const_cast<MVPolynomial*>(this)->operator[](index);
  }
  const mapped_type& operator[](key_type&& index) const {
    return const_cast<MVPolynomial*>(this)->operator[](index);
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

  MVPolynomial operator+() const { return *this; }

  MVPolynomial operator-() const {
    auto m = MVPolynomial(*this);
    for (auto& i_and_v : m) {
      auto& [_, v] = i_and_v;
      v            = -v;
    }
    return m;
  }

  MVPolynomial& operator*=(mapped_type r) {
    for (auto& i_and_v : index2value_) {
      auto& [_, v] = i_and_v;
      v *= r;
    }
    return *this;
  }

  // friend functions
  // TODO consider tolerance.
  friend bool operator==(const MVPolynomial& l, const MVPolynomial& r) {
    return l.index2value_ == r.index2value_;
  }

  // TODO consider tolerance.
  friend bool operator!=(const MVPolynomial& l, const MVPolynomial& r) { return !(l == r); }

  friend MVPolynomial operator+(const MVPolynomial& l, const MVPolynomial& r) {
    auto comparer = l.key_comp();
    auto sum      = MVPolynomial(comparer, l.get_allocator());
    sum.reserve(l.size() + r.size());
    auto l_it = l.begin();
    auto r_it = r.begin();
    // Like Merge sort algorithm, insert or sum data to sum.
    while (l_it != l.end() && r_it != r.end()) {
      const auto& [l_idx, l_v] = *l_it;
      const auto& [r_idx, r_v] = *r_it;
      if (comparer(l_idx, r_idx)) {
        sum.emplace_hint(sum.end(), *l_it);
        ++l_it;
      } else if (comparer(r_idx, l_idx)) {
        sum.emplace_hint(sum.end(), *r_it);
        ++r_it;
      } else {
        sum.emplace_hint(sum.end(), l_idx, l_v + r_v);
        ++l_it;
        ++r_it;
      }
    }
    // If r or l iterator don't equals its end, insert the extra to sum.
    while (l_it != l.end()) {
      sum.emplace_hint(sum.end(), *l_it);
      ++l_it;
    }
    while (r_it != r.end()) {
      sum.emplace_hint(sum.end(), *r_it);
      ++r_it;
    }
    return sum;
  }

  friend MVPolynomial operator-(const MVPolynomial& l, const MVPolynomial& r) {
    auto comparer = l.key_comp();
    auto sub      = MVPolynomial(comparer, l.get_allocator());
    sub.reserve(l.size() + r.size());
    auto l_it = l.begin();
    auto r_it = r.begin();
    // Like Merge sort algorithm, insert or subtract data to sub.
    while (l_it != l.end() && r_it != r.end()) {
      const auto& [l_idx, l_v] = *l_it;
      const auto& [r_idx, r_v] = *r_it;
      if (comparer(l_idx, r_idx)) {
        sub.emplace_hint(sub.end(), *l_it);
        ++l_it;
      } else if (comparer(r_idx, l_idx)) {
        sub.emplace_hint(sub.end(), r_idx, -r_v);
        ++r_it;
      } else {
        sub.emplace_hint(sub.end(), l_idx, l_v - r_v);
        ++l_it;
        ++r_it;
      }
    }
    // If r or l iterator don't equals its end, insert the extra to sub.
    while (l_it != l.end()) {
      sub.emplace_hint(sub.end(), *l_it);
      ++l_it;
    }
    while (r_it != r.end()) {
      const auto& [r_idx, r_v] = *r_it;
      sub.emplace_hint(sub.end(), r_idx, -r_v);
      ++r_it;
    }
    return sub;
  }

  friend MVPolynomial operator*(const MVPolynomial& l, const MVPolynomial& r) {
    auto comparer = l.key_comp();
    auto mul      = std::vector<value_type>();
    mul.reserve(l.size() * r.size());
    // Calculate all product of each l's term and r's term.
    for (const auto& l_p : l) {
      const auto& [l_idx, l_v] = l_p;
      for (const auto& r_p : r) {
        const auto& [r_idx, r_v] = r_p;
        mul.emplace_back(l_idx + r_idx, l_v * r_v);
      }
    }
    std::sort(mul.begin(), mul.end(), [&comparer](const value_type& l, const value_type& r) {
      return comparer(l.first, r.first);
    });
    auto rm_begin =
        std::unique(mul.begin(), mul.end(), [](const value_type& l, const value_type& r) {
          return (l.first == r.first).all();
        });
    // For simplicity, I make a MVPolynomial now.
    auto mp = MVPolynomial(
        boost::container::ordered_unique_range_t(),
        mul.begin(),
        rm_begin,
        comparer,
        l.get_allocator()
    );
    // Add duplicated elements to mp.
    for (auto it = rm_begin; it != mul.end(); ++it) {
      mp[it->first] += it->second;
    }

    return mp;
  }

  friend void swap(MVPolynomial& l, MVPolynomial& r) { swap(l.index2value_, r.index2value_); }

 private:
  void CheckSelfIndexes() const {
    // The last index is the lowest index of all index,
    // so I only have to check if each of its elements is non-negative.
    if ((index2value_.back().first < 0).any()) {
      throw std::invalid_argument(fmt::format("Negative index not supported!"));
    }
  }

  std::pair<iterator, bool> CheckIndexOfPairOfIterAndIsInserted(
      const std::pair<iterator, bool>& iter_and_is_inserted
  ) {
    const auto& [iter, is_inserted] = iter_and_is_inserted;
    if (is_inserted) {
      CheckIndex(iter->first);
    }
    return iter_and_is_inserted;
  }

  IndexContainer index2value_{
      {index_type::Zero(), 0}
  };
};

template <
    std::signed_integral IntType,
    class R,
    int D,
    class AllocatorOrContainer =
        boost::container::new_allocator<std::pair<IndexType<IntType, D>, R>>>
using DefaultMVPolynomial =
    MVPolynomial<IntType, R, D, IndexComparer<IntType, D>, AllocatorOrContainer>;

template <
    std::signed_integral IntType,
    class R,
    int Dim,
    class Comparer,
    class AllocatorOrContainer =
        boost::container::new_allocator<std::pair<IndexType<IntType, Dim>, R>>>
auto D(const MVPolynomial<IntType, R, Dim, Comparer, AllocatorOrContainer>& p, std::size_t axis) {
  using MP = MVPolynomial<IntType, R, Dim, Comparer, AllocatorOrContainer>;

  MP::CheckAxis(axis);

  auto new_index2value_seq = typename MP::sequence_type(p.get_allocator());
  new_index2value_seq.reserve(p.size());
  for (auto index_and_value : p) {
    auto [index, value] = index_and_value;
    if (index[axis] == 0) {
      continue;
    } else {
      value *= index[axis]--;
      new_index2value_seq.emplace_back(index, value);
    }
  }
  const auto& comparer = p.key_comp();
  // Sort indexes in order not to violate the order by comparer.
  std::sort(
      new_index2value_seq.begin(),
      new_index2value_seq.end(),
      [&comparer](const MP::value_type& l, const MP::value_type& r) {
        return comparer(l.first, r.first);
      }
  );
  return MP(
      boost::container::ordered_unique_range_t(),
      new_index2value_seq.begin(),
      new_index2value_seq.end(),
      comparer,
      p.get_allocator()
  );
}

template <
    std::signed_integral IntType,
    class R,
    int Dim,
    class AllocatorOrContainer =
        boost::container::new_allocator<std::pair<IndexType<IntType, Dim>, R>>>
auto D(const DefaultMVPolynomial<IntType, R, Dim, AllocatorOrContainer>& p, std::size_t axis) {
  using MP = DefaultMVPolynomial<IntType, R, Dim, AllocatorOrContainer>;

  MP::CheckAxis(axis);

  auto new_index2value_seq = typename MP::sequence_type(p.get_allocator());
  new_index2value_seq.reserve(p.size());
  auto p_it = p.begin();
  while (p_it != p.end()) {
    auto [index, value] = *p_it;
    if (index[axis] == 0) {
      if (axis == MP::dim - 1) {
        ++p_it;
      }
      auto d_end_it = p.end();
      for (std::size_t ith_axis = 0; ith_axis <= axis; ++ith_axis) {
        d_end_it = std::partition_point(
            p_it,
            d_end_it,
            [ith_axis, &index](const typename MP::value_type& v) {
              return v.first[ith_axis] == index[ith_axis];
            }
        );
      }
      // Skip indexes which axis-th element is zero.
      p_it = d_end_it;
    } else {
      value *= index[axis]--;
      new_index2value_seq.emplace_back(index, value);
      ++p_it;
    }
  }
  return MP(
      boost::container::ordered_unique_range_t(),
      new_index2value_seq.begin(),
      new_index2value_seq.end(),
      p.key_comp(),
      p.get_allocator()
  );
}

template <
    std::signed_integral IntType,
    class R,
    int D,
    class Comparer = IndexComparer<IntType, D>,
    class AllocatorOrContainer =
        boost::container::new_allocator<std::pair<IndexType<IntType, D>, R>>>
auto Integrate(MVPolynomial<IntType, R, D, Comparer, AllocatorOrContainer>&& p, std::size_t axis) {
  using MP = MVPolynomial<IntType, R, D, Comparer, AllocatorOrContainer>;

  MP::CheckAxis(axis);

  auto index2value = p.extract_sequence();
  for (auto& index_and_value : index2value) {
    auto& [index, value] = index_and_value;
    value /= ++index[axis];
  }
  const auto& comparer = p.key_comp();
  std::sort(
      index2value.begin(),
      index2value.end(),
      [&comparer](const typename MP::value_type& l, const typename MP::value_type& r) {
        return comparer(l.first, r.first);
      }
  );
  p.adopt_sequence(std::move(index2value));
  return std::move(p);
}

template <
    std::signed_integral IntType,
    class R,
    int D,
    class Comparer = IndexComparer<IntType, D>,
    class AllocatorOrContainer =
        boost::container::new_allocator<std::pair<IndexType<IntType, D>, R>>>
auto Integrate(
    const MVPolynomial<IntType, R, D, Comparer, AllocatorOrContainer>& p, std::size_t axis
) {
  return Integrate(MVPolynomial<IntType, R, D, Comparer, AllocatorOrContainer>(p), axis);
}

template <
    std::signed_integral IntType,
    class R,
    int D,
    class Comparer,
    class AllocatorOrContainer =
        boost::container::new_allocator<std::pair<IndexType<IntType, D>, R>>>
auto Of(
    const MVPolynomial<IntType, R, D, Comparer, AllocatorOrContainer>&                      p,
    const typename MVPolynomial<IntType, R, D, Comparer, AllocatorOrContainer>::coord_type& x
) {
  using MP                     = MVPolynomial<IntType, R, D, Comparer, AllocatorOrContainer>;
  typename MP::mapped_type sum = 0;
  for (const auto& index_and_value : p) {
    const auto& [index, value] = index_and_value;
    sum += value * (x.array().pow(index.template cast<typename MP::mapped_type>())).prod();
  }
  return sum;
}

template <class Iterator, class Coord>
auto OfImpl(Iterator begin, Iterator end, int dim, std::size_t axis, const Coord& x) {
  assert(axis >= 0 && axis < dim);
  if (axis == dim - 1) {
    auto [last_index, last_coeff] = *begin;
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
      const auto& [p_index, p_coeff] = *partition_point;
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

template <
    std::signed_integral IntType,
    class R,
    int D,
    class AllocatorOrContainer =
        boost::container::new_allocator<std::pair<IndexType<IntType, D>, R>>>
auto Of(
    const DefaultMVPolynomial<IntType, R, D, AllocatorOrContainer>&                      p,
    const typename DefaultMVPolynomial<IntType, R, D, AllocatorOrContainer>::coord_type& x
) {
  using MP = DefaultMVPolynomial<IntType, R, D, AllocatorOrContainer>;
  return OfImpl(p.cbegin(), p.cend(), MP::dim, 0, x);
}

template <template <class, int> class Array, class IntType, int D>
auto MakeSubIndex(const Array<IntType, D>& index) {
  auto projected_index = Array<IntType, D - 1>();
  std::for_each(
      boost::make_zip_iterator(boost::make_tuple(projected_index.begin(), index.cbegin())),
      boost::make_zip_iterator(boost::make_tuple(projected_index.end(), std::prev(index.cend()))),
      [](const auto& t) { boost::tuples::get<0>(t) = boost::tuples::get<1>(t); }
  );
  return projected_index;
}

template <
    std::signed_integral IntType,
    class R,
    int D,
    class AllocatorOrContainer =
        boost::container::new_allocator<std::pair<IndexType<IntType, D>, R>>>
class ExactOf {
 public:
  static_assert(D > 2);

  static const int dim{D};

  using alloc_traits    = std::allocator_traits<AllocatorOrContainer>;
  using polynomial_type = DefaultMVPolynomial<IntType, R, dim, AllocatorOrContainer>;
  using partition_type  = std::vector<typename polynomial_type::const_iterator>;

  using projected_polynomial_alloc_type =
      typename alloc_traits::rebind_alloc<std::pair<IndexType<IntType, dim - 1>, R>>;
  using projected_polynomial_type =
      DefaultMVPolynomial<IntType, R, dim - 1, projected_polynomial_alloc_type>;

  explicit ExactOf(polynomial_type&& p) { set_polynomial(std::move(p)); }

  explicit ExactOf(const polynomial_type& p) : ExactOf(polynomial_type(p)) {}

  ExactOf()                                = default;
  ExactOf(const ExactOf& other)            = default;
  ExactOf& operator=(const ExactOf& other) = default;
  ExactOf(ExactOf&& other)                 = default;
  ExactOf& operator=(ExactOf&& other)      = default;
  virtual ~ExactOf()                       = default;

  auto operator()(const typename polynomial_type::coord_type& x) {
    for (auto partition_it = partition_.begin(); partition_it != std::prev(partition_.end());
         ++partition_it) {
      projection_.get_polynomial_coeff(MakeSubIndex<IndexType>((*partition_it)->first)) =
          Calculate(*partition_it, *(std::next(partition_it)), dim - 1, x);
    }
    return projection_(MakeSubIndex<CoordType>(x));
  }

  const auto& get_polynomial() const { return polynomial_; }

  auto& get_polynomial_coeff(const typename polynomial_type::index_type& x) {
    return polynomial_[x];
  }

  void set_polynomial(polynomial_type&& p) {
    polynomial_ = std::move(p);

    // Make a partition from polynomial_.
    std::array<partition_type, dim - 1> partitions;
    MakePartitions(polynomial_, partitions);
    partition_ = partitions.back();

    // Make a projected polynomial.
    auto projected_polynomial = projected_polynomial_type();
    projected_polynomial.clear();
    for (const auto& polynomial_const_it :
         std::ranges::subrange(partition_.begin(), std::prev(partition_.end()))) {
      projected_polynomial.emplace_hint(
          projected_polynomial.end(),
          MakeSubIndex<IndexType>(polynomial_const_it->first),
          0
      );
    }
    projection_.set_polynomial(std::move(projected_polynomial));
  }

  void set_polynomial(const polynomial_type& p) { set_polynomial(polynomial_type(p)); }

  auto move_polynomial() { return std::move(polynomial_); }

 private:
  void MakePartitions(
      const polynomial_type& polynomial, std::array<partition_type, dim - 1>& partitions
  ) {
    // Make a partition from the index at 0 of p.
    // example:
    //   index:
    //   0:  1 1 1 1 2 2 2 2
    //   1:  2 2 3 3 1 1 2 2
    //   partition:
    //   0:  0       1      2(end)
    ConstructPartition(partitions[0], polynomial.cbegin(), polynomial.cend(), 0);
    // Make the other partition in the previous range.
    // example:
    //   index:
    //   0:  1 1 1 1 2 2 2 2
    //   1:  2 2 3 3 1 1 2 3
    //   partition:
    //   0:  0       1       2(end)
    //   1:  0   1   2   3 4 5(end)
    for (auto it = std::next(partitions.begin()); it != partitions.end(); ++it) {
      const auto& partition = *std::prev(it);
      for (auto partition_it = partition.cbegin(); partition_it != std::prev(partition.cend());
           ++partition_it) {
        ConstructPartition(
            *it,
            *partition_it,
            *(std::next(partition_it)),
            std::distance(partitions.begin(), it)
        );
        it->pop_back();
      }
      it->push_back(polynomial.cend());
    }
  }

  void ConstructPartition(
      partition_type&                     partition,
      typename partition_type::value_type cbegin,
      typename partition_type::value_type cend,
      int                                 axis
  ) {
    while (true) {
      partition.push_back(cbegin);
      const auto& [first_index, _] = *cbegin;
      auto partition_point         = std::partition_point(
          cbegin,
          cend,
          [axis, &first_index](const typename decltype(cbegin)::value_type& pair) {
            return pair.first[axis] == first_index[axis];
          }
      );
      if (partition_point == cend) {
        partition.push_back(cend);
        // The calculation ends.
        break;
      }
      cbegin = partition_point;
    }
  }

  template <class Iterator>
  auto Calculate(
      Iterator cbegin, Iterator cend, int axis, const typename polynomial_type::coord_type& x
  ) const {
    auto [last_index, last_coeff] = *cbegin;
    std::for_each(
        std::next(cbegin),
        cend,
        [&last_coeff, &last_index, &x, axis](const typename polynomial_type::value_type& i_and_c) {
          const auto& [next_index, next_coeff] = i_and_c;
          last_coeff *= std::pow(x[axis], last_index[axis] - next_index[axis]);
          last_coeff += next_coeff;
          last_index = next_index;
        }
    );
    last_coeff *= std::pow(x[axis], last_index[axis]);
    return last_coeff;
  }

  polynomial_type                                             polynomial_;
  partition_type                                              partition_;
  ExactOf<IntType, R, D - 1, projected_polynomial_alloc_type> projection_;
};

template <std::signed_integral IntType, class R, class AllocatorOrContainer>
class ExactOf<IntType, R, 2, AllocatorOrContainer> {
 public:
  static const int dim{2};

  using alloc_traits    = std::allocator_traits<AllocatorOrContainer>;
  using polynomial_type = DefaultMVPolynomial<IntType, R, dim, AllocatorOrContainer>;
  using partition_type  = std::vector<typename polynomial_type::const_iterator>;

  using projected_polynomial_alloc_type =
      typename alloc_traits::rebind_alloc<std::pair<IntType, R>>;
  using projected_polynomial_type = DefaultPolynomial<IntType, R, projected_polynomial_alloc_type>;

  explicit ExactOf(polynomial_type&& p) { set_polynomial(std::move(p)); }

  explicit ExactOf(const polynomial_type& p) : ExactOf(polynomial_type(p)) {}

  ExactOf()                                = default;
  ExactOf(const ExactOf& other)            = default;
  ExactOf& operator=(const ExactOf& other) = default;
  ExactOf(ExactOf&& other)                 = default;
  ExactOf& operator=(ExactOf&& other)      = default;
  virtual ~ExactOf()                       = default;

  auto operator()(const typename polynomial_type::coord_type& x) {
    for (auto partition_it = partition_.begin(); partition_it != std::prev(partition_.end());
         ++partition_it) {
      projection_.get_polynomial_coeff((*partition_it)->first[0]) =
          Calculate(*partition_it, *(std::next(partition_it)), dim - 1, x);
    }
    return projection_(x[0]);
  }

  const auto& get_polynomial() const { return polynomial_; }

  auto& get_polynomial_coeff(const typename polynomial_type::index_type& x) {
    return polynomial_[x];
  }

  void set_polynomial(polynomial_type&& p) {
    polynomial_ = std::move(p);

    // Make a partition from polynomial_.
    std::array<partition_type, dim - 1> partitions;
    MakePartitions(polynomial_, partitions);
    partition_ = partitions.back();

    // Make a projected polynomial.
    auto projected_polynomial = projected_polynomial_type();
    projected_polynomial.clear();
    for (const auto& polynomial_const_it :
         std::ranges::subrange(partition_.begin(), std::prev(partition_.end()))) {
      projected_polynomial.emplace_hint(
          projected_polynomial.end(),
          polynomial_const_it->first[0],
          polynomial_const_it->second
      );
    }
    projection_.set_polynomial(std::move(projected_polynomial));
  }

  void set_polynomial(const polynomial_type& p) { set_polynomial(polynomial_type(p)); }

  auto move_polynomial() { return std::move(polynomial_); }

 private:
  void MakePartitions(
      const polynomial_type& polynomial, std::array<partition_type, dim - 1>& partitions
  ) {
    // Make a partition from the index at 0 of p.
    // example:
    //   index:
    //   0:  1 1 1 1 2 2 2 2
    //   1:  2 2 3 3 1 1 2 2
    //   partition:
    //   0:  0       1      2(end)
    ConstructPartition(partitions[0], polynomial.cbegin(), polynomial.cend(), 0);
  }

  void ConstructPartition(
      partition_type&                     partition,
      typename partition_type::value_type cbegin,
      typename partition_type::value_type cend,
      int                                 axis
  ) {
    while (true) {
      partition.push_back(cbegin);
      const auto& [first_index, _] = *cbegin;
      auto partition_point         = std::partition_point(
          cbegin,
          cend,
          [axis, &first_index](const typename decltype(cbegin)::value_type& pair) {
            return pair.first[axis] == first_index[axis];
          }
      );
      if (partition_point == cend) {
        partition.push_back(cend);
        // The calculation ends.
        break;
      }
      cbegin = partition_point;
    }
  }

  template <class Iterator>
  auto Calculate(
      Iterator cbegin, Iterator cend, int axis, const typename polynomial_type::coord_type& x
  ) const {
    auto [last_index, last_coeff] = *cbegin;
    std::for_each(
        std::next(cbegin),
        cend,
        [&last_coeff, &last_index, &x, axis](const typename polynomial_type::value_type& i_and_c) {
          const auto& [next_index, next_coeff] = i_and_c;
          last_coeff *= std::pow(x[axis], last_index[axis] - next_index[axis]);
          last_coeff += next_coeff;
          last_index = next_index;
        }
    );
    last_coeff *= std::pow(x[axis], last_index[axis]);
    return last_coeff;
  }

  polynomial_type                                         polynomial_;
  partition_type                                          partition_;
  ExactOf<IntType, R, 1, projected_polynomial_alloc_type> projection_;
};

template <std::signed_integral IntType, class R, class AllocatorOrContainer>
class ExactOf<IntType, R, 1, AllocatorOrContainer> {
 public:
  static const int dim{1};

  using alloc_traits    = std::allocator_traits<AllocatorOrContainer>;
  using polynomial_type = Polynomial<IntType, R, IndexComparer<IntType, dim>, AllocatorOrContainer>;

  explicit ExactOf(polynomial_type&& p) { set_polynomial(std::move(p)); }

  explicit ExactOf(const polynomial_type& p) : ExactOf(polynomial_type(p)) {}

  ExactOf()                                = default;
  ExactOf(const ExactOf& other)            = default;
  ExactOf& operator=(const ExactOf& other) = default;
  ExactOf(ExactOf&& other)                 = default;
  ExactOf& operator=(ExactOf&& other)      = default;
  virtual ~ExactOf()                       = default;

  auto operator()(typename polynomial_type::coord_type x) const {
    auto cbegin                   = polynomial_.cbegin();
    auto cend                     = polynomial_.cend();
    auto [last_index, last_coeff] = *cbegin;
    std::for_each(
        std::next(cbegin),
        cend,
        [&last_coeff, &last_index, x](const typename polynomial_type::value_type& i_and_c) {
          const auto& [next_index, next_coeff] = i_and_c;
          last_coeff *= std::pow(x, last_index - next_index);
          last_coeff += next_coeff;
          last_index = next_index;
        }
    );
    last_coeff *= std::pow(x, last_index);
    return last_coeff;
  }

  const auto& get_polynomial() const { return polynomial_; }

  auto& get_polynomial_coeff(typename polynomial_type::index_type x) { return polynomial_[x]; }

  void set_polynomial(polynomial_type&& p) { polynomial_ = std::move(p); }

  void set_polynomial(const polynomial_type& p) { set_polynomial(polynomial_type(p)); }

  auto move_polynomial() { return std::move(polynomial_); }

 private:
  polynomial_type polynomial_;
};
}  // namespace mvPolynomial

#endif
