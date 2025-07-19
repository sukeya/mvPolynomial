# multivariable_polynomial
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
git clone https://github.com/sukeya/multivariable_polynomial.git
```
Next, add the following codes in your CMakeLists.txt.
```
add_subdirectory(multivariable_polynomial)
```
Finally, add linked libraries.
```
target_link_libraries(your_exe PRIVATE mvpolynomial)
```
Then, you will be able to use this library.

# How to use
The namespace is `mvpolynomial`.
The examples exist in "test" directory.

## Multi-variable polynomial
A class `MultiVarPolynomial` implements multi-variable polynomials.
Its interface is the same as Boost's flat_map except that it lacks `merge` member function.
So, please see the document of Boost's flat_map.

A class `ExactOf` calculates f of x as less multiplication as possible, but uses more memories.

## Polynomial product
A class `PolynomialProduct` implements a product of polynomials which have different variable each other (example: (1 + x + x^2) * (y + 6 * y^3 + y^10) * (1 + z)).
Its interface is the same as std::array.
So, please see the document of std::array.

## Polynomial
A class `Polynomial` implements one-variable polynomials.
Its interface is also the same as Boost's flat_map except that it lacks `merge` member function.
So, please see the document of Boost's flat_map.
