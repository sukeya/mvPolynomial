#ifndef _MVPOLYNOMIAL_TYPE_HPP_
#define _MVPOLYNOMIAL_TYPE_HPP_

#include <concepts>

#include "Eigen/Core"

namespace mvpolynomial {
template <std::signed_integral IntType, int D>
using IndexType = Eigen::Array<IntType, D, 1>;

template <std::floating_point R, int D>
using CoordType = Eigen::Array<R, D, 1>;
}  // namespace mvpolynomial

#endif
