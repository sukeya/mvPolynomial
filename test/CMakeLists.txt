FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        b3fb4b9feafcd8d91c5cb510a4775143fdbef02f # v3.11.0
  OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(Catch2)

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
    Catch2::Catch2WithMain
)

add_test(NAME mvPolynomial_test COMMAND mvPolynomial_test)
