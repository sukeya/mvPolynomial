find_package(Boost REQUIRED COMPONENTS unit_test_framework)

add_library(mvPolynomial_test_lib INTERFACE)
target_link_libraries(
  mvPolynomial_test_lib
  INTERFACE
    ${PROJECT_NAME}
    Boost::unit_test_framework
)
target_compile_options(mvPolynomial_test_lib INTERFACE
    "$<$<CONFIG:Debug>:-g;-O0;-Wall;-Wextra>"
    "$<$<CONFIG:Release>:-DNDEBUG;-O3;-march=native;-mtune=native>"
)

add_executable(mvPolynomial_test mvPolynomial_test.cpp)
target_link_libraries(
  mvPolynomial_test
  PRIVATE
    mvPolynomial_test_lib
)
add_test(NAME mvPolynomial_test COMMAND mvPolynomial_test)


add_executable(polynomial_product_test polynomial_product_test.cpp)
target_link_libraries(
  polynomial_product_test
  PRIVATE
    mvPolynomial_test_lib
)
add_test(NAME polynomial_product_test COMMAND polynomial_product_test)


add_executable(polynomial_test polynomial_test.cpp)
target_link_libraries(
  polynomial_test
  PRIVATE
    mvPolynomial_test_lib
)
add_test(NAME polynomial_test COMMAND polynomial_test)
