# mvPolynomial
This is an implementation of multi-variable polynomial in C++.

# Idea
This library treats multi-variable polynomial as a map from indexes to coefficients.
For example, `x^2 * y^2 - 2 * x * y ^2 + 3` is
```
{
  {{2, 2}, 1},
  {{1, 2}, -2},
  {{0, 0}, 3}
}
```

# Install
First, you clone this repository.
```
git clone https://github.com/sukeya/mvPolynomial.git
```
Next, add the following codes in your CMakeLists.txt.
```
add_subdirectory(mvPolynomial)
```
Finally, add linked libraries.
```
target_link_libraries(your_exe PRIVATE mvPolynomial)
```
Then, you will be able to use this library.

# How to use
The namespace is `mvPolynomial`.
The examples exist in "test" directory.

## Multi-variable polynomial
A class `MVPolynomial` implements multi-variable polynomials.

```cpp
#include "mvPolynomial/mvPolynomial.hpp"

using MP2 = mvPolynomial::MVPolynomial<int, double, 2>;
```

You can construct a polynomial from an initializer list of `(index, coefficient)` pairs.

```cpp
auto p = MP2({
    {{0, 0}, 1.0},
    {{1, 0}, 2.0},
    {{0, 1}, 3.0},
});
```

The constant term is the zero index.

```cpp
double c = p.get(Eigen::Array2i::Zero());
p.set(Eigen::Array2i({2, 0}), 5.0);
```

`set(index, 0.0)` removes the term, and zero-coefficient terms are normalized away automatically.

The class provides basic polynomial arithmetic.

```cpp
auto q   = MP2({
    {{0, 0}, 4.0},
    {{1, 0}, 1.0},
});
auto sum = p + q;
auto sub = p - q;
auto mul = p * q;
auto pw  = p.pow(3);
```

You can evaluate a polynomial at a point.

```cpp
double value = p(Eigen::Vector2d({2.0, 3.0}));
```

You can also substitute a polynomial for one axis.

```cpp
auto x = MP2({
    {{0, 0}, 1.0},
    {{1, 0}, 2.0},
});
auto composed = p(x, 1);
```

Differentiation and integration are free functions.

```cpp
auto dx = mvPolynomial::D(p, 0);
auto iy = mvPolynomial::Integrate(p, 1);
```

Negative indices are not supported and cause `std::invalid_argument`.
