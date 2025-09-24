#ifndef _MVPOLYNOMIAL_POLYNOMIAL_EXPRESSION_HPP_
#define _MVPOLYNOMIAL_POLYNOMIAL_EXPRESSION_HPP_

#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace myPolynomial {
template <class T>
bool is_const_ref_v = std::is_const_v<T> && std::is_reference_v<T>;

template <class T>
bool is_rvalue_v = std::is_rvalue_reference_v<T> && (!std::is_const_v<T>);

template <class T>
struct FilterRvalue;

template <class T>
struct FilterRvalue<T&&> {
  using type = T;

  static constexpr bool is_rvalue = true;
};

template <class T>
struct FilterRvalue<const T&> {
  using type = const T&;

  static constexpr bool is_rvalue = false;
};

template <class T>
requires is_const_ref_v<T> || is_rvalue_v<T>
class RefWrapper {
  using FilterResult = FilterRvalue<T>;

 public:
  using Domain = typename T::Domain;
  using Range  = typename T::Range;

  using Storage = FilterResult::value;

  static constexpr bool is_rvalue = FilterResult::is_rvalue;

  RefWrapper(T&& t) : t_(std::forward<T>(t)) {}

  Range operator()(const Domain& x) const { return t_(x); }

  const Storage& read() const
  requires(!is_rvalue)
  {
    return t_;
  }
  const Storage& read() const
  requires(is_rvalue)
  {
    return t_;
  }

  const Storage& move() &&
  requires(!is_rvalue)
  {
    return t_;
  }
  Storage move() &&
  requires(is_rvalue)
  {
    return std::move(t_);
  }

 private:
  Storage t_;
};

// L and R may be either const& or &&.
template <class L, class Op, class R>
requires(is_const_ref_v<L> || is_rvalue_v<L>) && (is_const_ref_v<R> || is_rvalue_v<R>)
class BinaryExpr {
  using LRef = RefWrapper<L>;
  using RRef = RefWrapper<L>;

 public:
  static_assert(std::is_same_v<typename L::Domain, typename R::Domain>, "Domain mismatch!");
  static_assert(std::is_same_v<typename L::Range, typename R::Range>, "Range mismatch!");

  using Domain = typename L::Domain;
  using Range  = typename L::Range;

  static constexpr bool is_l_rvalue = LRef::is_rvalue;
  static constexpr bool is_r_rvalue = RRef::is_rvalue;

  BinaryExpr(L&& l, R&& r) : l_(std::forward<L>(l)), r_(std::forward<R>(r)) {}

  // Calculate x of this
  Range operator()(const Domain& x) const { return Op::Apply(l_(x), r_(x)); }

  const auto& read_l() const { return l_.read(); }

  const auto& read_r() const { return r_.read(); }

  auto move_l()
  requires(is_l_rvalue)
  {
    return std::move(l_).move();
  }

  decltype(auto) move_l()
  requires(!is_l_rvalue)
  {
    return std::move(l_).move();
  }

  auto move_r()
  requires(is_r_rvalue)
  {
    return std::move(r_).move();
  }

  decltype(auto) move_r()
  requires(!is_r_rvalue)
  {
    return std::move(r_).move();
  }

 private:
  LRef l_;
  RRef r_;
};

struct Plus {
  template <class R>
  requires(!std::is_floating_point_v<R>)
  static R Apply(R&& l, R&& r) {
    return std::forward<R>(l) + std::forward<R>(r);
  }

  template <class R>
  requires std::is_floating_point_v<R>
  static R Apply(R l, R r) {
    return l + r;
  }
};

struct Minus {
  template <class R>
  requires(!std::is_floating_point_v<R>)
  static R Apply(R&& l, R&& r) {
    return std::forward<R>(l) - std::forward<R>(r);
  }

  template <class R>
  requires std::is_floating_point_v<R>
  static R Apply(R l, R r) {
    return l - r;
  }
};

struct Multiply {
  template <class R>
  requires(!std::is_floating_point_v<R>)
  static R Apply(R&& l, R&& r) {
    return std::forward<R>(l) * std::forward<R>(r);
  }

  template <class R>
  requires std::is_floating_point_v<R>
  static R Apply(R l, R r) {
    return l * r;
  }
};

// F and G may be either const& or &&.
template <class F, class G>
requires(is_const_ref_v<F> || is_rvalue_v<F>) && (is_const_ref_v<G> || is_rvalue_v<G>)
class Composition {
  using FRef = RefWrapper<F>;
  using GRef = RefWrapper<G>;

 public:
  static_assert(std::is_same_v<typename F::Domain, typename G::Range>, "Cannot composite!");

  using Domain = typename G::Domain;
  using Range  = typename F::Range;

  static constexpr bool is_outer_rvalue = FRef::is_rvalue;
  static constexpr bool is_inner_rvalue = GRef::is_rvalue;

  Composition(F&& f, G&& g) : f_(std::forward<F>(f)), g_(std::forward<G>(g)) {}

  Range operator()(const Domain& x) { return f_(g_(x)); }

  const auto& read_outer() const { return f_.read(); }

  const auto& read_inner() const { return g_.read(); }

  auto move_outer()
  requires(is_outer_rvalue)
  {
    return std::move(f_).move();
  }

  decltype(auto) move_outer()
  requires(!is_outer_rvalue)
  {
    return std::move(f_).move();
  }

  auto move_inner()
  requires(is_inner_rvalue)
  {
    return std::move(g_).move();
  }

  decltype(auto) move_inner()
  requires(!is_inner_rvalue)
  {
    return std::move(g_).move();
  }

 private:
  FRef f_;
  GRef g_;
};

template <class L, class R>
auto D(int axis, BinaryExpr<L, Plus, R>&& expr) {
  auto d_l = D(axis, expr.move_l());
  auto d_r = D(axis, expr.move_r());
  return BinaryExpr<decltype(d_l)&&, Plus, decltype(d_r)&&>{std::move(d_l), std::move(d_r)};
}

template <class L, class R>
auto D(int axis, const BinaryExpr<L, Plus, R>& expr) {
  auto d_l = D(axis, expr.read_l());
  auto d_r = D(axis, expr.read_r());
  return BinaryExpr<decltype(d_l)&&, Plus, decltype(d_r)&&>{std::move(d_l), std::move(d_r)};
}

template <class L, class R>
auto D(int axis, BinaryExpr<L, Minus, R>&& expr) {
  auto d_l = D(axis, expr.move_l());
  auto d_r = D(axis, expr.move_r());
  return BinaryExpr<decltype(d_l)&&, Minus, decltype(d_r)&&>{std::move(d_l), std::move(d_r)};
}

template <class L, class R>
auto D(int axis, const BinaryExpr<L, Minus, R>& expr) {
  auto d_l = D(axis, expr.read_l());
  auto d_r = D(axis, expr.read_r());
  return BinaryExpr<decltype(d_l)&&, Minus, decltype(d_r)&&>{std::move(d_l), std::move(d_r)};
}

template <class L, class R, class DL, class DR>
auto DMultiplyImpl(L&& l, R&& r, DL&& d_l, DR&& d_r) {
  auto l_prod =
      BinaryExpr<decltype(d_l)&&, Multiply, decltype(r)>{std::move(d_l), std::forward<R>(r)};
  auto r_prod =
      BinaryExpr<decltype(l), Multiply, decltype(d_r)&&>{std::forward<R>(l), std::move(d_r)};
  return BinaryExpr<decltype(l_prod)&&, Plus, decltype(r_prod)&&>{
      std::move(l_prod),
      std::move(r_prod)
  };
}

template <class L, class R>
auto D(int axis, BinaryExpr<L, Multiply, R>&& expr) {
  auto d_l = D(axis, expr.read_l());
  auto d_r = D(axis, expr.read_r());
  return DMultiplyImpl(expr.move_l(), expr.move_r(), std::move(d_l), std::move(d_r));
}

template <class L, class R>
auto D(int axis, const BinaryExpr<L, Multiply, R>& expr) {
  auto d_l = D(axis, expr.read_l());
  auto d_r = D(axis, expr.read_r());
  return DMultiplyImpl(expr.read_l(), expr.read_r(), std::move(d_l), std::move(d_r));
}

template <class F, class G>
auto D(int axis, Composition<F, G>&& comp) {
  auto        d_f = D(axis, comp.move_outer());
  const auto& g   = comp.read_inner();
  auto        d_g = D(axis, g);
  return BinaryExpr<Composition<decltype(d_f)&&, G&&>, Multiply, decltype(d_g)&&>{
      Composition{std::move(d_f), comp.move_inner()},
      std::move(d_g)
  };
}

template <class F, class G>
auto D(int axis, const Composition<F, G>& comp) {
  auto        d_f = D(axis, comp.read_outer());
  const auto& g   = comp.read_inner();
  auto        d_g = D(axis, g);
  return BinaryExpr<Composition<decltype(d_f)&&, const G&>, Multiply, decltype(d_g)&&>{
      Composition{std::move(d_f), g},
      std::move(d_g)
  };
}

template <class B, class E, class L, class R>
auto S(std::size_t axis, B&& begin, E&& end, BinaryExpr<L, Plus, R>&& expr) {
  auto s_l = S(axis, std::forward<B>(begin), std::forward<E>(end), expr.move_l());
  auto s_r = S(axis, std::forward<B>(begin), std::forward<E>(end), expr.move_r());
  return BinaryExpr<decltype(s_l)&&, Plus, decltype(s_r)&&>{std::move(s_l), std::move(s_r)};
}

template <class B, class E, class L, class R>
auto S(std::size_t axis, B&& begin, E&& end, const BinaryExpr<L, Plus, R>& expr) {
  auto s_l = S(axis, std::forward<B>(begin), std::forward<E>(end), expr.read_l());
  auto s_r = S(axis, std::forward<B>(begin), std::forward<E>(end), expr.read_r());
  return BinaryExpr<decltype(s_l)&&, Plus, decltype(s_r)&&>{std::move(s_l), std::move(s_r)};
}

template <class B, class E, class L, class R>
auto S(std::size_t axis, B&& begin, E&& end, BinaryExpr<L, Minus, R>&& expr) {
  auto s_l = S(axis, std::forward<B>(begin), std::forward<E>(end), expr.move_l());
  auto s_r = S(axis, std::forward<B>(begin), std::forward<E>(end), expr.move_r());
  return BinaryExpr<decltype(s_l)&&, Minus, decltype(s_r)&&>{std::move(s_l), std::move(s_r)};
}

template <class B, class E, class L, class R>
auto S(std::size_t axis, B&& begin, E&& end, const BinaryExpr<L, Minus, R>& expr) {
  auto s_l = S(axis, std::forward<B>(begin), std::forward<E>(end), expr.read_l());
  auto s_r = S(axis, std::forward<B>(begin), std::forward<E>(end), expr.read_r());
  return BinaryExpr<decltype(s_l)&&, Minus, decltype(s_r)&&>{std::move(s_l), std::move(s_r)};
}
}  // namespace myPolynomial

#endif
