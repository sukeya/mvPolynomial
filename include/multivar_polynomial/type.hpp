#ifndef _MULTIVAR_POLYNOMIAL_TYPE_HPP_
#define _MULTIVAR_POLYNOMIAL_TYPE_HPP_

#include <concepts>

#include "Eigen/Core"

namespace multivar_polynomial {
template <std::signed_integral IntType, int D>
using IndexType = Eigen::Array<IntType, D, 1>;

template <class R, int D>
using CoordType = Eigen::Array<R, D, 1>;
}  // namespace multivar_polynomial

#endif
