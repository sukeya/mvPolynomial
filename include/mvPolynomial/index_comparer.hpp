#ifndef _MVPOLYNOMIAL_INDEX_COMPARER_HPP_
#define _MVPOLYNOMIAL_INDEX_COMPARER_HPP_

#include "mvPolynomial/type.hpp"

#include <concepts>
#include <compare>
#include <cstddef>

namespace mvPolynomial {
/**
 * \brief A class comparing two multivariable polynomials by its indeces.
 * \tparam IntType the type of elements of indices.
 * \tparam D the dimension of indices.
 */
template <std::signed_integral Int_, int D>
class IndexComparer final {
 public:
  static_assert(D > 0, "IndexComparer: the dimension must be positive.");

  using Int   = Int_;
  using Index = IndexType<Int, D>;

  static constexpr int dim = D;

  /**
   * \brief If lhd[i] == rhd[i] for i = 0, ..., N - 1 and lhd[N] > rhd[N],
   * return greater: if lhd[N] < rhd[N], return less: otherwise, return equal.
   * \param[in] lhd an index
   * \param[in] rhd an index
   */
  static constexpr std::strong_ordering Compare(const Index& lhd, const Index& rhd) {
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

  /**
   * \brief If lhd[i] == rhd[i] for i = 0, ..., N - 1 and lhd[N] > rhd[N],
   * return true: otherwise, false.
   * \param[in] lhd an index
   * \param[in] rhd an index
   */
  static constexpr bool IsGreater(const Index& lhd, const Index& rhd) {
    for (std::size_t i = 0; i != lhd.size(); ++i) {
      if (lhd[i] > rhd[i]) {
        return true;
      }
    }
    return false;
  }
};
}  // namespace mvPolynomial

#endif
