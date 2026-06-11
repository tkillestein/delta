cmake_minimum_required(VERSION 3.18)
project(delta LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Standards / defaults
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---------------------------------------------------------------------------
# Python + nanobind
# ---------------------------------------------------------------------------
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

# Locate nanobind's CMake config via the installed package.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# ---------------------------------------------------------------------------
# Core numeric / IO dependencies
#   Eigen, CFITSIO and FFTW are resolved via pkg-config (robust across versions).
#   CFITSIO/FFTW are wired now but only exercised from later milestones (M5/M7).
# ---------------------------------------------------------------------------
find_package(PkgConfig REQUIRED)
pkg_check_modules(EIGEN REQUIRED IMPORTED_TARGET eigen3)
pkg_check_modules(CFITSIO IMPORTED_TARGET cfitsio)
pkg_check_modules(FFTWF IMPORTED_TARGET fftw3f)
find_package(OpenMP)

# ---------------------------------------------------------------------------
# Python extension module
# ---------------------------------------------------------------------------
nanobind_add_module(_core src/bindings.cpp)
target_include_directories(_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(_core PRIVATE PkgConfig::EIGEN)
if(OpenMP_CXX_FOUND)
  target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)
endif()

install(TARGETS _core LIBRARY DESTINATION delta)
