cmake_minimum_required(VERSION 3.25.2)

# Set vcpkg toolchain.
set(
    CMAKE_TOOLCHAIN_FILE
    "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake"
    CACHE STRING "Vcpkg toolchain file"
)

project(multivar_polynomial VERSION 0.1.0 LANGUAGES CXX)

# Config
set(CMAKE_CXX_EXTENSIONS OFF)
add_library(compiler_flags INTERFACE)
target_compile_features(compiler_flags INTERFACE cxx_std_20)
target_compile_options(compiler_flags INTERFACE
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
find_package(Boost REQUIRED COMPONENTS unit_test_framework)
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
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/>
)
target_link_libraries(
  ${PROJECT_NAME}
  INTERFACE
    ${extra_libs}
    compiler_flags
)

add_subdirectory(test)
