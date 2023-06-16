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
    $<BUILD_INTERFACE:${extra_headers}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(
  ${PROJECT_NAME}
  INTERFACE
    ${extra_libs}
    multivar_polynomial_compiler_flags
)

add_subdirectory(test)

install(
  TARGETS
    ${PROJECT_NAME}
    multivar_polynomial_compiler_flags
  EXPORT
    multivar_polynomial_targets
  DESTINATION
    lib
)

install(
  FILES
    include/multivar_polynomial/multivar_polynomial.hpp
    include/multivar_polynomial/polynomial_product.hpp
  DESTINATION
    include/multivar_polynomial
)

install(
  EXPORT
    multivar_polynomial_targets
  FILE
    multivar_polynomial_targets.cmake
  DESTINATION
    lib/cmake/multivar_polynomial
)

include(CMakePackageConfigHelpers)
# generate the config file that includes the exports
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/multivar_polynomial-config.cmake"
  INSTALL_DESTINATION "lib/cmake/multivar_polynomial"
  NO_SET_AND_CHECK_MACRO
  NO_CHECK_REQUIRED_COMPONENTS_MACRO
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/multivar_polynomial-config-version.cmake"
  VERSION "${CMAKE_PROJECT_VERSION_MAJOR}.${CMAKE_PROJECT_VERSION_MINOR}"
  COMPATIBILITY SameMinorVersion
)
install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/multivar_polynomial-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/multivar_polynomial-config-version.cmake
  DESTINATION
    lib/cmake/multivar_polynomial
)
export(
  EXPORT multivar_polynomial_targets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/multivar_polynomial_targets.cmake"
)

include(InstallRequiredSystemLibraries)
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_VERSION_MAJOR "${CMAKE_PROJECT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${CMAKE_PROJECT_VERSION_MINOR}")
set(CPACK_SOURCE_GENERATOR "TGZ")
include(CPack)
