cmake_minimum_required(VERSION 3.30)

project(mvPolynomial VERSION 0.1.0 LANGUAGES CXX)

# Config
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_OPTIMIZE_DEPENDENCIES ON)

# Library
if (MVPOLYNOMIAL_BUILD_TESTS)
  FetchContent_Declare(
    eigen3
    GIT_REPOSITORY https://github.com/boostorg/test.git
    GIT_TAG        cfd3a7e8749752f5a9d20266c1fc7afdb9b1e173 # 5.0.0
    OVERRIDE_FIND_PACKAGE
  )
  FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG        e424e3f2e607da02742f73db84873b8084fc714c # 12.0.0
    OVERRIDE_FIND_PACKAGE
  )
  FetchContent_MakeAvailable(eigen3 fmt)
else
  find_package(Eigen3 5.0.0 CONFIG REQUIRED)
  find_package(fmt 12.0.0 CONFIG REQUIRED)
endif()

list(APPEND extra_libs Eigen3::Eigen)
list(APPEND extra_libs fmt::fmt)

# To avoid add_subdirectory,
FetchContent_Declare(
  platanus
  GIT_REPOSITORY https://github.com/boostorg/test.git
  GIT_TAG        origin/main # TODO Fix commit hash
)
list(APPEND extra_libs platanus)


add_library(${PROJECT_NAME} INTERFACE)
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_20)
target_include_directories(
  ${PROJECT_NAME}
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(
  ${PROJECT_NAME}
  INTERFACE
    ${extra_libs}
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(gcc "$<COMPILE_LANG_AND_ID:CXX,GNU>")
set(clang "$<COMPILE_LANG_AND_ID:CXX,Clang>")
set(msvc "$<COMPILE_LANG_AND_ID:CXX,MSVC>")

set(gcc_like $<OR:${gcc},${clang})

if (MVPOLYNOMIAL_BUILD_TESTS)
  enable_testing()

  FetchContent_Declare(
    boost_unit_test
    GIT_REPOSITORY https://github.com/boostorg/test.git
    GIT_TAG        cfd3a7e8749752f5a9d20266c1fc7afdb9b1e173 # boost-1.89.0
    OVERRIDE_FIND_PACKAGE
  )
  FetchContent_MakeAvailable(boost_unit_test)

  add_subdirectory(test)
endif()
