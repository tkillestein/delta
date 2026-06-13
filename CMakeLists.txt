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
# Architecture tuning
#   The convolution / variance / score inner loops are written to auto-vectorise;
#   the default x86-64 baseline only guarantees SSE2, leaving AVX/AVX2/AVX-512
#   on the table. DELTA_NATIVE (default ON) builds for the host's full ISA, which
#   is the right choice for a from-source `uv sync` install and typically a 2-4x
#   speedup on the separable/dense kernels. Turn it OFF for a portable wheel.
#   We deliberately do NOT enable -ffast-math: the pipeline relies on IEEE NaN/inf
#   semantics (std::isfinite masking), which fast-math would break.
# ---------------------------------------------------------------------------
option(DELTA_NATIVE "Tune the compiled core for the host CPU (-march=native)" ON)

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
#   FFTW is wired now but only exercised from a later milestone (M7).
# ---------------------------------------------------------------------------
find_package(PkgConfig REQUIRED)
pkg_check_modules(EIGEN REQUIRED IMPORTED_TARGET eigen3)
pkg_check_modules(CFITSIO REQUIRED IMPORTED_TARGET cfitsio)
pkg_check_modules(FFTWF REQUIRED IMPORTED_TARGET fftw3f)
find_package(OpenMP)

# ---------------------------------------------------------------------------
# Python extension module
# ---------------------------------------------------------------------------
nanobind_add_module(_core
  src/bindings.cpp
  src/io.cpp
  src/basis.cpp
  src/convolve.cpp
  src/detect.cpp
  src/spatial.cpp
  src/solve.cpp
  src/subtract.cpp
  src/fit.cpp
  src/noise.cpp
)
target_include_directories(_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(_core PRIVATE PkgConfig::EIGEN PkgConfig::CFITSIO PkgConfig::FFTWF)
if(OpenMP_CXX_FOUND)
  target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)
endif()

# nanobind's module helper appends -Os (optimise for size) to keep the binary
# small; for a numerics core that silently disables the vectorisation/unrolling
# -O3 enables and cripples the convolution/variance/score hot loops. Re-assert
# -O3 *after* nanobind's flags (last -O wins) so the kernels actually vectorise.
target_compile_options(_core PRIVATE -O3)

if(DELTA_NATIVE AND (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang"))
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-march=native" DELTA_HAS_MARCH_NATIVE)
  if(DELTA_HAS_MARCH_NATIVE)
    target_compile_options(_core PRIVATE -march=native)
  endif()
endif()

install(TARGETS _core LIBRARY DESTINATION delta)
