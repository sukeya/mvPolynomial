cmake_minimum_required(VERSION 3.25.2)

project(multivar_polynomial VERSION 0.1.0 LANGUAGES CXX)

# Config
set(CMAKE_CXX_EXTENSIONS OFF)
add_library(multivar_polynomial_compiler_flags INTERFACE)
target_compile_features(multivar_polynomial_compiler_flags INTERFACE cxx_std_20)
target_compile_options(multivar_polynomial_compiler_flags INTERFACE
    "$<$<CONFIG:Debug>:-g;-O0;-Wall;-Wextra>"
    "$<$<CONFIG:Release>:-DNDEBUG;-O3;-march=native;-mtune=native>"
)
set(CMAKE_OPTIMIZE_DEPENDENCIES ON)


# Library
# Boost
set(Boost_USE_STATIC_LIBS        ON)  # only find static libs
set(Boost_USE_DEBUG_LIBS        OFF)  # ignore debug libs and
set(Boost_USE_RELEASE_LIBS       ON)  # only find release libs
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME    OFF)
list(APPEND extra_headers ${Boost_INCLUDE_DIRS})

# Eigen
find_package(Eigen3 CONFIG REQUIRED NO_MODULE)
list(APPEND extra_libs Eigen3::Eigen)

# fmt
find_package(fmt CONFIG REQUIRED)
list(APPEND extra_libs fmt::fmt)


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
    multivar_polynomial_compiler_flags
)

add_subdirectory(test)
