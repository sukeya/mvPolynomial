#ifndef _MULTIVAR_POLYNOMIAL_POLYNOMIAL_HPP_
#define _MULTIVAR_POLYNOMIAL_POLYNOMIAL_HPP_


#include "multivar_polynomial/type.hpp"
#include "multivar_polynomial/index_comparer.hpp"

#include <concepts>

#include "boost/container/flat_map.hpp"
#include "boost/container/new_allocator.hpp"
#include "fmt/core.h"


namespace multivar_polynomial
{
  template <
    class R,
    std::signed_integral IntType,
    class Comparer = IndexComparer<IntType, 1>,
    class AllocatorOrContainer = boost::container::new_allocator<std::pair<IntType, R>> 
  >
  class Polynomial
  {
  public:
    inline static const int dim{1};

    using index_type = IntType;
    using coord_type = R;

  private:
    using IndexContainer = boost::container::flat_map<index_type, R, Comparer, AllocatorOrContainer>;

  public:
    using key_type = IndexContainer::key_type;
    using mapped_type = IndexContainer::mapped_type;
    using value_type = IndexContainer::value_type;
    using movable_value_type = IndexContainer::movable_value_type;

    using key_compare = IndexContainer::key_compare;
    using value_compare = IndexContainer::value_compare;

    using sequence_type = IndexContainer::sequence_type;

    using allocator_type = IndexContainer::allocator_type;
    using allocator_traits_type = IndexContainer::allocator_traits_type;
    using stored_allocator_type = IndexContainer::stored_allocator_type;

    using pointer = IndexContainer::pointer;
    using const_pointer = IndexContainer::const_pointer;

    using reference = IndexContainer::reference;
    using const_reference = IndexContainer::const_reference;

    using size_type = IndexContainer::size_type;
    using difference_type = IndexContainer::difference_type;

    using iterator = IndexContainer::iterator;
    using const_iterator = IndexContainer::const_iterator;
    using reverse_iterator = IndexContainer::reverse_iterator;
    using const_reverse_iterator = IndexContainer::const_reverse_iterator;


    explicit Polynomial(const allocator_type& allocator) : index2value_(allocator)
    {
      CheckSelfIndexes();
    }

    explicit Polynomial(const Comparer& comparer) : index2value_(comparer) { CheckSelfIndexes(); }

    Polynomial(const Comparer& comparer, const allocator_type& allocator)
     : index2value_(comparer, allocator) { CheckSelfIndexes(); }

    template<typename InputIterator>
    Polynomial(InputIterator s, InputIterator e) : index2value_(s, e) { CheckSelfIndexes(); }

    template<typename InputIterator> 
    explicit Polynomial(InputIterator s, InputIterator e, const allocator_type& allocator)
     : index2value_(s, e, allocator) { CheckSelfIndexes(); }

    template<typename InputIterator> 
    explicit Polynomial(InputIterator s, InputIterator e, const Comparer& c)
     : index2value_(s, e, c) { CheckSelfIndexes(); }

    template<typename InputIterator> 
    explicit Polynomial(InputIterator s, InputIterator e, const Comparer& c, const allocator_type& a)
     : index2value_(s, e, c, a) { CheckSelfIndexes(); }

    template<typename InputIterator> 
    explicit Polynomial(boost::container::ordered_unique_range_t o, InputIterator s, InputIterator e)
     : index2value_(o, s, e) { CheckSelfIndexes(); }

    template<typename InputIterator> 
    explicit Polynomial(
      boost::container::ordered_unique_range_t o, InputIterator s, InputIterator e, const Comparer& c
    ) : index2value_(o, s, e, c) { CheckSelfIndexes(); }

    template<typename InputIterator> 
    explicit Polynomial(
      boost::container::ordered_unique_range_t o,
      InputIterator s,
      InputIterator e, 
      const Comparer& c,
      const allocator_type& a
    ) : index2value_(o, s, e, c, a) { CheckSelfIndexes(); }

    template<typename InputIterator> 
    explicit Polynomial(
      boost::container::ordered_unique_range_t o,
      InputIterator s,
      InputIterator e, 
      const allocator_type& a
    ) : index2value_(o, s, e, a) { CheckSelfIndexes(); }

    explicit Polynomial(std::initializer_list<value_type> l) : index2value_(l) { CheckSelfIndexes(); }

    explicit Polynomial(std::initializer_list<value_type> l, const allocator_type& a)
     : index2value_(l, a) { CheckSelfIndexes(); }

    explicit Polynomial(std::initializer_list<value_type> l, const Comparer& c)
     : index2value_(l, c) { CheckSelfIndexes(); }

    explicit Polynomial(std::initializer_list<value_type> l, const Comparer& c, const allocator_type& a)
     : index2value_(l, c, a) { CheckSelfIndexes(); }

    explicit Polynomial(boost::container::ordered_unique_range_t o, std::initializer_list<value_type> l)
     : index2value_(o, l) { CheckSelfIndexes(); }

    explicit Polynomial(
      boost::container::ordered_unique_range_t o,
      std::initializer_list<value_type> l,
      const Comparer& c
    ) : index2value_(o, l, c) { CheckSelfIndexes(); }

    explicit Polynomial(
      boost::container::ordered_unique_range_t o,
      std::initializer_list<value_type> l,
      const Comparer& c,
      const allocator_type& a
    ) : index2value_(o, l, c, a) { CheckSelfIndexes(); }

    explicit Polynomial(const Polynomial& m, const allocator_type& a)
     : index2value_(m.index2value_, a) { CheckSelfIndexes(); }

    explicit Polynomial(Polynomial&& m, const allocator_type& a)
     : index2value_(std::move(m.index2value_), a) { CheckSelfIndexes(); }

    Polynomial& operator=(std::initializer_list<value_type> l)
    {
      index2value_ = l;
      CheckSelfIndexes();
      return *this;
    }

    explicit Polynomial(mapped_type r)
    {
      index2value_.at(0) = r;
    }

    Polynomial() = default;
    Polynomial(const Polynomial& other) = default;
    Polynomial& operator=(const Polynomial& other) = default;
    Polynomial(Polynomial&& other) = default;
    Polynomial& operator=(Polynomial&& other) = default;
    virtual ~Polynomial() = default;


    allocator_type get_allocator() const noexcept { return index2value_.get_allocator(); }

    auto get_stored_allocator() noexcept { return index2value_.get_stored_allocator(); }
    auto get_stored_allocator() const noexcept { return index2value_.get_stored_allocator(); }

    iterator begin() noexcept { return index2value_.begin(); }
    const_iterator begin() const noexcept  { return index2value_.begin(); }

    iterator end() noexcept { return index2value_.end(); }
    const_iterator end() const noexcept { return index2value_.end(); }

    reverse_iterator rbegin() noexcept { return index2value_.rbegin(); }
    const_reverse_iterator rbegin() const noexcept { return index2value_.rbegin(); }

    reverse_iterator rend() noexcept { return index2value_.rend(); }
    const_reverse_iterator rend() const noexcept { return index2value_.rend(); }
    
    const_iterator cbegin() const noexcept { return index2value_.cbegin(); }
    const_iterator cend() const noexcept { return index2value_.cend(); }

    const_reverse_iterator crbegin() const noexcept { return index2value_.crbegin(); }
    const_reverse_iterator crend() const noexcept { return index2value_.crend(); }

    bool empty() const noexcept { return index2value_.empty(); }

    size_type size() const noexcept { return index2value_.size(); }

    size_type max_size() const noexcept { return index2value_.max_size(); }

    size_type capacity() const noexcept { return index2value_.capacity(); }

    void reserve(size_type size) { return index2value_.reserve(size); }

    void shrink_to_fit() { return index2value_.shrink_to_fit(); }

    mapped_type& operator[](const key_type& index) { return index2value_[index]; }
    mapped_type& operator[](key_type&& index) { return index2value_[index]; }

    const mapped_type& operator[](const key_type& index) const { return const_cast<Polynomial*>(this)->operator[](index); }
    const mapped_type& operator[](key_type&& index) const { return const_cast<Polynomial*>(this)->operator[](index); }

    template<typename M> 
    std::pair<iterator, bool> insert_or_assign(const key_type& i, M&& m)
    {
      CheckIndex(i);
      return index2value_.insert_or_assign(i, std::move(m));
    }

    template<typename M> 
    std::pair<iterator, bool> insert_or_assign(key_type&& i, M&& m)
    {
      CheckIndex(i);
      return index2value_.insert_or_assign(std::move(i), std::move(m));
    }

    template<typename M> 
    iterator insert_or_assign(const_iterator ci, const key_type& i, M&& m)
    {
      CheckIndex(i);
      return index2value_.insert_or_assign(ci, i, std::move(m));
    }

    template<typename M> 
    iterator insert_or_assign(const_iterator ci, key_type&& i, M&& m)
    {
      CheckIndex(i);
      return index2value_.insert_or_assign(ci, std::move(i), std::move(m));
    }

    iterator nth(size_type size) noexcept { return index2value_.nth(size); }
    const_iterator nth(size_type size) const noexcept { return index2value_.nth(size); }

    size_type index_of(iterator i) noexcept { return index2value_.index_of(i); }
    size_type index_of(const_iterator ci) const noexcept { return index2value_.index_of(ci); }

    mapped_type& at(const key_type& i) { return index2value_.at(i); }
    const mapped_type& at(const key_type& i) const { return index2value_.at(i); }

    template<class... Args>
    std::pair<iterator, bool> emplace(Args &&... args)
    {
      return CheckIndexOfPairOfIterAndIsInserted(index2value_.emplace(std::move(args...)));
    }

    template<class... Args>
    iterator emplace_hint(const_iterator ci, Args &&... args)
    {
      auto iter = index2value_.emplace_hint(ci, std::move(args)...);
      CheckIndex(iter->first);
      return iter;
    }

    template<class... Args> 
    std::pair<iterator, bool> try_emplace(const key_type& i, Args &&... args)
    {
      return CheckIndexOfPairOfIterAndIsInserted(index2value_.try_emplace(i, std::move(args...)));
    }

    template<class... Args> 
    iterator try_emplace(const_iterator ci, const key_type& i, Args &&... args)
    {
      auto iter = index2value_.try_emplace(ci, i, std::move(args...));
      CheckIndex(iter->first);
      return iter;
    }

    template<class... Args> 
    std::pair<iterator, bool> try_emplace(key_type&& i, Args &&... args)
    {
      return CheckIndexOfPairOfIterAndIsInserted(index2value_.try_emplace(std::move(i), std::move(args...)));
    }

    template<class... Args> 
    iterator try_emplace(const_iterator ci, key_type&& i, Args &&... args)
    {
      auto iter = index2value_.try_emplace(ci, std::move(i), std::move(args...));
      CheckIndex(iter->first);
      return iter;
    }

    std::pair<iterator, bool> insert(const value_type& i_and_v)
    {
      return CheckIndexOfPairOfIterAndIsInserted(index2value_.insert(i_and_v));
    }

    std::pair<iterator, bool> insert(value_type&& i_and_v)
    {
      return CheckIndexOfPairOfIterAndIsInserted(index2value_.insert(std::move(i_and_v)));
    }

    template<typename Pair> 
    std::pair<iterator BOOST_MOVE_I bool> insert(Pair&& p)
    {
      return CheckIndexOfPairOfIterAndIsInserted(index2value_.insert(std::move(p)));
    }

    iterator insert(const_iterator ci, const value_type& i_and_v)
    {
      auto iter = index2value_.insert(ci, i_and_v);
      CheckIndex(iter->first);
      return iter;
    }

    iterator insert(const_iterator ci, value_type&& i_and_v)
    {
      auto iter = index2value_.insert(ci, std::move(i_and_v));
      CheckIndex(iter->first);
      return iter;
    }

    template<class Pair>
    iterator insert(const_iterator ci, Pair&& p)
    {
      auto iter = index2value_.insert(ci, std::move(p));
      CheckIndex(iter->first);
      return iter;
    }

    template<typename InputIterator>
    void insert(InputIterator s, InputIterator e)
    {
      index2value_.insert(s, e);
      CheckSelfIndexes();
    }

    template<typename InputIterator> 
    void insert(boost::container::ordered_unique_range_t o, InputIterator s, InputIterator e)
    {
      index2value_.insert(o, s, e);
      CheckSelfIndexes();
    }

    void insert(std::initializer_list<value_type> l)
    {
      index2value_.insert(l);
      CheckSelfIndexes();
    }

    void insert(boost::container::ordered_unique_range_t o, std::initializer_list<value_type> l)
    {
      index2value_.insert(o, l);
      CheckSelfIndexes();
    }

    iterator erase(const_iterator ci) { return index2value_.erase(ci); }
    size_type erase(const value_type& i_and_v) { return index2value_.erase(i_and_v); }
    iterator erase(const_iterator s, const_iterator e) { return index2value_.erase(s, e); }

    void swap(Polynomial& m) { index2value_.swap(m.index2value_); }

    void clear() noexcept { index2value_.clear(); }

    key_compare key_comp() const { return index2value_.key_comp(); }
    value_compare value_comp() const  { return index2value_.value_comp(); }

    iterator find(const key_type& i) { return index2value_.find(i); }
    const_iterator find(const key_type& i) const { return index2value_.find(i); }
    template<typename K>
    iterator find(const K& i) { return index2value_.find(i); }
    template<typename K>
    const_iterator find(const K& i) const { return index2value_.find(i); }

    size_type count(const key_type& i) const { return index2value_.count(i); }
    template<typename K>
    size_type count(const K& i) const { return index2value_.count(i); }

    bool contains(const key_type& i) const { return index2value_.contains(i); }
    template<typename K>
    bool contains(const K& i) const { return index2value_.contains(i); }

    iterator lower_bound(const key_type& i) { return index2value_.lower_bound(i); }
    const_iterator lower_bound(const key_type& i) const { return index2value_.lower_bound(i); }
    template<typename K>
    iterator lower_bound(const K& i) { return index2value_.lower_bound(i); }
    template<typename K>
    const_iterator lower_bound(const K& i) const { return index2value_.lower_bound(i); }

    iterator upper_bound(const key_type& i) { return index2value_.upper_bound(i); }
    const_iterator upper_bound(const key_type& i) const { return index2value_.upper_bound(i); }
    template<typename K>
    iterator upper_bound(const K& i) { return index2value_.upper_bound(i); }
    template<typename K>
    const_iterator upper_bound(const K& i) const { return index2value_.upper_bound(i); }

    std::pair<iterator, iterator> equal_range(const key_type& i)
    {
      return index2value_.equal_range(i);
    }

    std::pair<const_iterator, const_iterator> equal_range(const key_type& i) const
    {
      return index2value_.equal_range(i);
    }

    template<typename K>
    std::pair<iterator, iterator> equal_range(const K& i)
    {
      return index2value_.equal_range(i);
    }

    template<typename K> 
    std::pair<const_iterator, const_iterator> equal_range(const K& i) const
    {
      return index2value_.equal_range(i);
    }

    sequence_type extract_sequence() { return index2value_.extract_sequence(); }

    void adopt_sequence(sequence_type&& seq) { index2value_.adopt_sequence(std::move(seq)); }
    void adopt_sequence(boost::container::ordered_unique_range_t o, sequence_type&& seq)
    {
      index2value_.adopt_sequence(o, std::move(seq));
    }

    const sequence_type& sequence() const noexcept { return index2value_.sequence(); }


    reference front() { return *(index2value_.begin()); }
    const_reference front() const { return *(index2value_.cbegin()); }

    reference back() { return *(index2value_.rbegin()); }
    const_reference back() const { return *(index2value_.crbegin()); }


    Polynomial operator+() const { return *this; }

    Polynomial operator-() const
    {
      auto m = Polynomial(*this);
      for (auto& i_and_v : m)
      {
        auto& [i, v] = i_and_v;
        v = -v;
      }
      return m;
    }

    Polynomial& operator*=(mapped_type r)
    {
      for(auto& i_and_v : index2value_)
      {
        auto& [i, v] = i_and_v;
        v *= r;
      }
      return *this;
    }

    // friend functions
    // TODO consider tolerance.
    friend bool operator==(const Polynomial& l, const Polynomial& r)
    {
      return l.index2value_ == r.index2value_;
    }

    // TODO consider tolerance.
    friend bool operator!=(const Polynomial& l, const Polynomial& r)
    {
      return l.index2value_ != r.index2value_;
    }

    friend Polynomial operator+(const Polynomial& l, const Polynomial& r)
    {
      auto comparer = l.key_comp();
      auto sum = Polynomial(comparer, l.get_allocator());
      sum.reserve(l.size() + r.size());
      auto l_it = l.begin();
      auto r_it = r.begin();
      // Like Merge sort algorithm, insert or sum data to sum.
      while(l_it != l.end() && r_it != r.end())
      {
        auto [l_idx, l_v] = *l_it;
        auto [r_idx, r_v] = *r_it;
        if(comparer(l_idx, r_idx))
        {
          sum.emplace_hint(sum.end(), *l_it);
          ++l_it;
        }
        else if(comparer(r_idx, l_idx))
        {
          sum.emplace_hint(sum.end(), *r_it);
          ++r_it;
        }
        else
        {
          sum.emplace_hint(sum.end(), l_idx, l_v + r_v);
          ++l_it;
          ++r_it;
        }
      }
      // If r or l iterator don't equals its end, insert the extra to sum.
      while(l_it != l.end())
      {
        sum.emplace_hint(sum.end(), *l_it);
        ++l_it;
      }
      while(r_it != r.end())
      {
        sum.emplace_hint(sum.end(), *r_it);
        ++r_it;
      }
      return sum;
    }

    friend Polynomial operator-(const Polynomial& l, const Polynomial& r)
    {
      auto comparer = l.key_comp();
      auto sub = Polynomial(comparer, l.get_allocator());
      sub.reserve(l.size() + r.size());
      auto l_it = l.begin();
      auto r_it = r.begin();
      // Like Merge sort algorithm, insert or subtract data to sub.
      while(l_it != l.end() && r_it != r.end())
      {
        auto [l_idx, l_v] = *l_it;
        auto [r_idx, r_v] = *r_it;
        if(comparer(l_idx, r_idx))
        {
          sub.emplace_hint(sub.end(), *l_it);
          ++l_it;
        }
        else if(comparer(r_idx, l_idx))
        {
          sub.emplace_hint(sub.end(), r_idx, -r_v);
          ++r_it;
        }
        else
        {
          sub.emplace_hint(sub.end(), l_idx, l_v - r_v);
          ++l_it;
          ++r_it;
        }
      }
      // If r or l iterator don't equals its end, insert the extra to sub.
      while(l_it != l.end())
      {
        sub.emplace_hint(sub.end(), *l_it);
        ++l_it;
      }
      while(r_it != r.end())
      {
        const auto& [r_idx, r_v] = *r_it;
        sub.emplace_hint(sub.end(), r_idx, -r_v);
        ++r_it;
      }
      return sub;
    }

    friend Polynomial operator*(const Polynomial& l, const Polynomial& r)
    {
      auto comparer = l.key_comp();
      auto mul = std::vector<value_type>();
      mul.reserve(l.size() * r.size());
      // Calculate all product of each l's term and r's term.
      for(const auto& l_p : l)
      {
        const auto& [l_idx, l_v] = l_p;
        for(const auto& r_p : r)
        {
          const auto& [r_idx, r_v] = r_p;
          mul.emplace_back(l_idx + r_idx, l_v * r_v);
        }
      }
      std::sort(
        mul.begin(),
        mul.end(),
        [&comparer](const value_type& l, const value_type& r) { return comparer(l.first, r.first); }
      );
      auto rm_begin = std::unique(
        mul.begin(),
        mul.end(),
        [](const value_type& l, const value_type& r) { return (l.first == r.first).all(); }
      );
      // For simplicity, I make a Polynomial now.
      auto mp = Polynomial(
        boost::container::ordered_unique_range_t(),
        mul.begin(),
        rm_begin,
        comparer,
        l.get_allocator()
      );
      // Add duplicated elements to mp.
      for(auto it = rm_begin; it != mul.end(); ++it)
      {
        mp[it->first] += it->second;
      }

      return mp;
    }

    friend void swap(Polynomial& l, Polynomial& r)
    {
      swap(l.index2value_, r.index2value_);
    }

  private:
    void CheckIndex(key_type index) const
    {
      if (index < 0)
      {
        throw std::runtime_error(
          fmt::format("The index ({}) must be non-negative.", index)
        );
      }
    }

    void CheckSelfIndexes() const
    {
      for (const auto& index_and_value : index2value_)
      {
        const auto& [index, value] = index_and_value;
        CheckIndex(index);
      }
    }

    std::pair<iterator, bool> CheckIndexOfPairOfIterAndIsInserted(const std::pair<iterator, bool>& iter_and_is_inserted)
    {
      const auto& [iter, is_inserted] = iter_and_is_inserted;
      if(is_inserted)
      {
        CheckIndex(iter->first);
      } 
      return iter_and_is_inserted;
    }

    IndexContainer index2value_{{0, 0}};
  };


  template <
    class R,
    std::signed_integral IntType,
    class Comparer,
    class AllocatorOrContainer = boost::container::new_allocator<std::pair<IntType, R>> 
  >
  auto D(const Polynomial<R, IntType, Comparer, AllocatorOrContainer>& p)
  {
    using MP = Polynomial<R, IntType, Comparer, AllocatorOrContainer>;

    auto new_index2value_seq = typename MP::sequence_type(p.get_allocator());
    new_index2value_seq.reserve(p.size());
    for(auto index_and_value : p)
    {
      auto [index, value] = index_and_value;
      if(index == 0)
      {
        continue;
      }
      else
      {
        value *= index--;
        new_index2value_seq.emplace_back(index, value);
      }
    }
    const auto& comparer = p.key_comp();
    // Sort indexes in order not to violate the order by comparer.
    std::sort(
      new_index2value_seq.begin(),
      new_index2value_seq.end(),
      [&comparer](const MP::value_type& l, const MP::value_type& r)
      {
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
    class R,
    std::signed_integral IntType,
    class AllocatorOrContainer = boost::container::new_allocator<std::pair<IntType, R>> 
  >
  auto D(Polynomial<R, IntType, IndexComparer<IntType, 1>, AllocatorOrContainer>&& p)
  {
    using MP = Polynomial<R, IntType, IndexComparer<IntType, 1>, AllocatorOrContainer>;

    auto index2value_seq = p.extract_sequence();
    if (index2value_seq.back().first == 0)
    {
      index2value_seq.pop_back();
    }
    for(auto& index_and_value : index2value_seq)
    {
      auto& [index, value] = index_and_value;
      value *= index--;
    }
    p.adopt_sequence(boost::container::ordered_unique_range_t(), std::move(index2value_seq));
    return p;
  }


  template <
    class R,
    std::signed_integral IntType,
    class AllocatorOrContainer = boost::container::new_allocator<std::pair<IntType, R>> 
  >
  auto D(const Polynomial<R, IntType, IndexComparer<IntType, 1>, AllocatorOrContainer>& p)
  {
    using MP = Polynomial<R, IntType, IndexComparer<IntType, 1>, AllocatorOrContainer>;
    return D(MP(p));
  }


  template <
    class R,
    std::signed_integral IntType,
    class Comparer = IndexComparer<IntType, 1>,
    class AllocatorOrContainer = boost::container::new_allocator<std::pair<IndexType<IntType, 1>, R>> 
  >
  auto Integrate(Polynomial<R, IntType, Comparer, AllocatorOrContainer>&& p)
  {
    using MP = Polynomial<R, IntType, Comparer, AllocatorOrContainer>;

    auto index2value = p.extract_sequence();
    for(auto& index_and_value : index2value)
    {
      auto& [index, value] = index_and_value;
      value /= ++index;
    }
    const auto& comparer = p.key_comp();
    std::sort(
      index2value.begin(),
      index2value.end(),
      [&comparer]
      (const typename MP::value_type& l, const typename MP::value_type& r){ return comparer(l.first, r.first); }
    );
    p.adopt_sequence(std::move(index2value));
    return std::move(p);
  }


  template <
    class R,
    std::signed_integral IntType,
    class Comparer = IndexComparer<IntType, 1>,
    class AllocatorOrContainer = boost::container::new_allocator<std::pair<IndexType<IntType, 1>, R>> 
  >
  auto Integrate(const Polynomial<R, IntType, Comparer, AllocatorOrContainer>& p)
  {
    return Integrate(Polynomial<R, IntType, Comparer, AllocatorOrContainer>(p));
  }


  template <
    class R,
    std::signed_integral IntType,
    class Comparer,
    class AllocatorOrContainer = boost::container::new_allocator<std::pair<IndexType<IntType, 1>, R>> 
  >
  auto Of(
    const Polynomial<R, IntType,  Comparer, AllocatorOrContainer>& p,
    const typename Polynomial<R, IntType, Comparer, AllocatorOrContainer>::coord_type& x
  )
  {
    using MP = Polynomial<R, IntType, Comparer, AllocatorOrContainer>;

    typename MP::mapped_type sum = 0;
    for(const auto& index_and_value : p)
    {
      const auto& [index, value] = index_and_value;
      sum += value * std::pow(x, index);
    }
    return sum;
  }


  template <
    class R,
    std::signed_integral IntType,
    class AllocatorOrContainer = boost::container::new_allocator<std::pair<IndexType<IntType, 1>, R>> 
  >
  auto Of(
    const Polynomial<R, IntType, IndexComparer<IntType, 1>, AllocatorOrContainer>& p,
    const typename Polynomial<R, IntType, IndexComparer<IntType, 1>, AllocatorOrContainer>::coord_type& x
  )
  {
    auto [last_index, last_coeff] = p.front();
    for(auto it = std::next(p.cbegin()); it != p.cend(); ++it)
    {
      const auto& [next_index, next_coeff] = *it;
      last_coeff *= std::pow(x, last_index - next_index);
      last_coeff += next_coeff;
      last_index = next_index;
    }
    last_coeff *= std::pow(x, last_index);
    return last_coeff;
  }
}

#endif