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
#   the plain x86-64 baseline only guarantees SSE2, leaving AVX/AVX2 on the
#   table. The default is therefore the **x86-64-v3 micro-architecture level**
#   (AVX2 + FMA + BMI2; Intel Haswell 2013+ / AMD Excavator 2015+) whenever the
#   compiler supports the flag -- a sensible floor for any machine running
#   survey-scale image subtraction. Non-x86 targets (e.g. arm64) skip it
#   automatically via the compile-flag check.
#
#   DELTA_X86_64_V3=OFF drops back to the fully portable SSE2 baseline for the
#   rare pre-2013 deployment. DELTA_NATIVE=ON supersedes both and builds for
#   the build host's full ISA (including AVX-512 where present) -- fastest, but
#   the binary SIGILLs on a different (older / differently-featured) CPU, so
#   keep it to from-source installs that only ever run on the build host.
#   We deliberately do NOT enable -ffast-math: the pipeline relies on IEEE NaN/inf
#   semantics (std::isfinite masking), which fast-math would break.
# ---------------------------------------------------------------------------
option(DELTA_NATIVE "Tune the compiled core for the host CPU (-march=native); non-portable" OFF)
option(DELTA_X86_64_V3
       "Compile for the x86-64-v3 level (AVX2/FMA, 2013+ CPUs) when supported; OFF = SSE2 baseline"
       ON)

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
# NOTE (perf): building with nanobind's LTO option was measured (paired
# min-of-7 benchmark runs) at +/-1.5% -- indistinguishable from run noise; the
# hot kernels are already header-visible within their TUs. Left off.
nanobind_add_module(_core
  src/bindings.cpp
  src/io.cpp
  src/basis.cpp
  src/convolve.cpp
  src/detect.cpp
  src/catalog.cpp
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

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  include(CheckCXXCompilerFlag)
  if(DELTA_NATIVE)
    check_cxx_compiler_flag("-march=native" DELTA_HAS_MARCH_NATIVE)
    if(DELTA_HAS_MARCH_NATIVE)
      target_compile_options(_core PRIVATE -march=native)
    endif()
  elseif(DELTA_X86_64_V3)
    check_cxx_compiler_flag("-march=x86-64-v3" DELTA_HAS_X86_64_V3)
    if(DELTA_HAS_X86_64_V3)
      target_compile_options(_core PRIVATE -march=x86-64-v3)
    endif()
  endif()
endif()

install(TARGETS _core LIBRARY DESTINATION delta)
