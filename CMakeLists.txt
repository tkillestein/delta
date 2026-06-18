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
#   on the table. DELTA_NATIVE builds for the host's full ISA, typically a 2-4x
#   speedup on the separable/dense kernels.
#
#   It defaults OFF so the produced binary is portable: a `-march=native` build
#   made on one host and run on a different (older, or differently-featured) CPU
#   SIGILLs. Turn it ON for a from-source install you will only ever run on the
#   build host (e.g. a local `uv sync`); leave it OFF for any redistributable
#   wheel or shared-cluster build.
#   We deliberately do NOT enable -ffast-math: the pipeline relies on IEEE NaN/inf
#   semantics (std::isfinite masking), which fast-math would break.
# ---------------------------------------------------------------------------
option(DELTA_NATIVE "Tune the compiled core for the host CPU (-march=native); non-portable" OFF)

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
#   Eigen and CFITSIO are resolved via pkg-config (robust across versions). The
#   FFT used by the noise-decorrelation pass is the vendored header-only PocketFFT
#   (extern/pocketfft, BSD-3) -- no external/system FFT dependency, and no GPL link
#   (PocketFFT is the same FFT NumPy/SciPy ship).
# ---------------------------------------------------------------------------
find_package(PkgConfig REQUIRED)
pkg_check_modules(EIGEN REQUIRED IMPORTED_TARGET eigen3>=3.4)
pkg_check_modules(CFITSIO REQUIRED IMPORTED_TARGET cfitsio)
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
  src/timing.cpp
)
target_include_directories(_core PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/extern/pocketfft)
target_link_libraries(_core PRIVATE PkgConfig::EIGEN PkgConfig::CFITSIO)
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
