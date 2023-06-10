include(CTest)


add_executable(multivar_polynomial_test multivar_polynomial_test.cpp)
target_link_libraries(
  multivar_polynomial_test
  PRIVATE
    ${PROJECT_NAME}
    Boost::unit_test_framework
)
add_test(NAME multivar_polynomial_test COMMAND multivar_polynomial_test)