#ifndef _MULTIVAR_POLYNOMIAL_INDEX_COMPARER_HPP_
#define _MULTIVAR_POLYNOMIAL_INDEX_COMPARER_HPP_

#include "multivar_polynomial/type.hpp"

#include <concepts>

namespace multivar_polynomial {
template <std::signed_integral IntType, int D>
class IndexComparer {
 public:
  static_assert(D > 1, "IndexComparer: the dimension must be greater than 1.");

  using Index = IndexType<IntType, D>;

  auto operator()(const Index& lhd, const Index& rhd) const {
    for (std::size_t i = 0; i != lhd.size(); ++i) {
      if (lhd[i] > rhd[i]) {
        return true;
      } else if (lhd[i] < rhd[i]) {
        return false;
      }
    }
    return false;
  }
};

template <std::signed_integral IntType>
class IndexComparer<IntType, 1> {
 public:
  using Index = IntType;

  auto operator()(Index lhd, Index rhd) const { return lhd > rhd; }
};
}  // namespace multivar_polynomial

#endif
