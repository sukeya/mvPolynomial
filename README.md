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
