cmake_minimum_required(VERSION 3.30)

project(mvPolynomial VERSION 0.1.0 LANGUAGES CXX)

# Config
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_OPTIMIZE_DEPENDENCIES ON)

# Library
# Boost
find_package(Boost CONFIG REQUIRED)
list(APPEND extra_headers ${Boost_INCLUDE_DIRS})

# Eigen
find_package(Eigen3 CONFIG REQUIRED NO_MODULE)
list(APPEND extra_libs Eigen3::Eigen)

# fmt
find_package(fmt CONFIG REQUIRED)
list(APPEND extra_libs fmt::fmt)

add_library(mvPolynomial_compiler_flags INTERFACE)
target_compile_features(mvPolynomial_compiler_flags INTERFACE cxx_std_20)

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(
  ${PROJECT_NAME}
  INTERFACE
    ${extra_headers}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(
  ${PROJECT_NAME}
  INTERFACE
    ${extra_libs}
    mvPolynomial_compiler_flags
)

if (BUILD_TESTS)
  include(CTest)
  add_subdirectory(test)
endif()
