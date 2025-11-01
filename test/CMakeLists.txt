add_executable(mvPolynomial_test mvPolynomial_test.cpp)
target_compile_options(
  mvPolynomial_test
  INTERFACE
    # Common options for gcc and clang
    "$<${gcc_like}>:-Wall;-Wextra>"
    "$<${msvc}:/Wall>"
)
target_link_libraries(
  mvPolynomial_test
  PRIVATE
    ${PROJECT_NAME}
    Boost::unit_test_framework
)

add_test(NAME mvPolynomial_test COMMAND mvPolynomial_test)
