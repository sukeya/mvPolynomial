find_package(Boost REQUIRED COMPONENTS unit_test_framework)

add_library(mvpolynomial_test_lib INTERFACE)
target_link_libraries(
  mvpolynomial_test_lib
  INTERFACE
    ${PROJECT_NAME}
    Boost::unit_test_framework
)
target_compile_options(mvpolynomial_test_lib INTERFACE
    "$<$<CONFIG:Debug>:-g;-O0;-Wall;-Wextra>"
    "$<$<CONFIG:Release>:-DNDEBUG;-O3;-march=native;-mtune=native>"
)

add_executable(mvpolynomial_test mvpolynomial_test.cpp)
target_link_libraries(
  mvpolynomial_test
  PRIVATE
    mvpolynomial_test_lib
)
add_test(NAME mvpolynomial_test COMMAND mvpolynomial_test)


add_executable(polynomial_product_test polynomial_product_test.cpp)
target_link_libraries(
  polynomial_product_test
  PRIVATE
    mvpolynomial_test_lib
)
add_test(NAME polynomial_product_test COMMAND polynomial_product_test)


add_executable(polynomial_test polynomial_test.cpp)
target_link_libraries(
  polynomial_test
  PRIVATE
    mvpolynomial_test_lib
)
add_test(NAME polynomial_test COMMAND polynomial_test)
