cmake_minimum_required(VERSION 3.30)

project(mvPolynomial VERSION 0.1.0 LANGUAGES CXX)

# Config
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_OPTIMIZE_DEPENDENCIES ON)

# Library
include(FetchContent)
if (MVPOLYNOMIAL_BUILD_TESTS)
  FetchContent_Declare(
    eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        549bf8c75b6aae071cde2f28aa48f16ee3ae60b0 # 5.0.0
    OVERRIDE_FIND_PACKAGE
  )
  # The reason why I use CACHE INTERNAL is written in the following link:
  # https://cmake.org/cmake/help/latest/module/FetchContent.html#overriding-where-to-find-cmakelists-txt
  set(EIGEN_BUILD_TESTING OFF CACHE INTERNAL "")
  set(EIGEN_BUILD_CMAKE_PACKAGE ON CACHE INTERNAL "")
  FetchContent_MakeAvailable(eigen3)

  FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG        e424e3f2e607da02742f73db84873b8084fc714c # 12.0.0
    OVERRIDE_FIND_PACKAGE
  )
  FetchContent_MakeAvailable(fmt)
else()
  find_package(Eigen3 5.0.0 CONFIG REQUIRED)
  find_package(fmt 12.0.0 CONFIG REQUIRED)
endif()

list(APPEND extra_libs Eigen3::Eigen)
list(APPEND extra_libs fmt::fmt)

# To avoid add_subdirectory,
FetchContent_Declare(
  platanus
  GIT_REPOSITORY https://github.com/sukeya/platanus.git
  GIT_TAG        0a0f3c93a637263c0479a398d54564a74c9ad694 # origin/main
)
FetchContent_MakeAvailable(platanus)
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

set(gcc "$<COMPILE_LANG_AND_ID:CXX,GNU>")
set(clang "$<COMPILE_LANG_AND_ID:CXX,Clang>")
set(msvc "$<COMPILE_LANG_AND_ID:CXX,MSVC>")

set(gcc_like $<OR:${gcc},${clang})

if (MVPOLYNOMIAL_BUILD_TESTS)
  enable_testing()

  add_subdirectory(test)
endif()
