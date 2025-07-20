#ifndef _MVPOLYNOMIAL_INDEX_COMPARER_HPP_
#define _MVPOLYNOMIAL_INDEX_COMPARER_HPP_

#include "mvPolynomial/type.hpp"

#include <concepts>
#include <compare>

namespace mvPolynomial {
/**
 * \brief A class comparing two multivariable polynomials by its indeces.
 * \tparam IntType the type of elements of indices.
 * \tparam D the dimension of indices.
 */
template <std::signed_integral IntType, int D>
class IndexComparer {
 public:
  static_assert(D > 0, "IndexComparer: the dimension must be positive.");

  using Index = IndexType<IntType, D>;

  /**
   * \brief If the elems of lhd are equal to that of rhd until i th time and lhd[i] is greater than
   * rhd[i], return true: otherwise, false.
   * \param[in] lhd an index
   * \param[in] rhd an index
   */
  constexpr bool operator()(const Index& lhd, const Index& rhd) const noexcept {
    for (std::size_t i = 0; i != lhd.size(); ++i) {
      auto comp = lhd[i] <=> rhd[i];
      if (comp > 0) {
        return true;
      } else if (comp < 0) {
        return false;
      }
    }
    return false;
  }

  constexpr std::strong_ordering get_ordering(const Index& lhd, const Index& rhd) const noexcept {
    for (std::size_t i = 0; i != lhd.size(); ++i) {
      auto comp = lhd[i] <=> rhd[i];
      if (comp > 0) {
        return std::strong_ordering::greater;
      } else if (comp < 0) {
        return std::strong_ordering::less;
      }
    }
    return std::strong_ordering::equal;
  }
};

template <std::signed_integral IntType>
class IndexComparer<IntType, 1> {
 public:
  using Index = IntType;

  constexpr bool operator()(Index lhd, Index rhd) const noexcept { return lhd > rhd; }
};
}  // namespace mvPolynomial

#endif
