#pragma once

// Opt-in, low-overhead stage timing for the C++ core.
//
// The Python pipeline already times the coarse stages (stamp selection, kernel
// solve, full-frame subtract, ...) via loguru. This adds finer *sub-stage*
// timers inside the two big C++ entry points (`fit_kernel`, `subtract`) so the
// `B_n` precompute/convolve cost can be split from the GLS solve and from the
// variance/mask passes (benchmarks/PERFORMANCE.md). It is gated on the
// `DELTA_TIMING` environment variable: when unset the timers compile to a single
// cached bool check and do nothing, so production runs pay no measurable cost.
//
// Usage:
//   delta::timing::clear();           // at the top of an instrumented call
//   { DELTA_TIME("fit: GLS solve"); ...work... }   // RAII scope
//   auto stages = delta::timing::drain();          // {label, seconds} in order

#include <chrono>
#include <string>
#include <utility>
#include <vector>

namespace delta::timing {

// True iff DELTA_TIMING is set to a non-empty, non-"0" value (cached on first
// call). When false, ScopedTimer is inert.
bool enabled();

// Accumulate `seconds` under `label` (thread-safe; preserves first-seen order).
void add(const std::string& label, double seconds);

// Reset the registry. Call before an instrumented top-level operation so its
// timings do not mix with a previous call's.
void clear();

// Snapshot the accumulated timings in first-seen order.
std::vector<std::pair<std::string, double>> drain();

// RAII timer: records its lifetime under `label` on destruction (no-op unless
// timing is enabled). Threads accumulate into the same label.
struct ScopedTimer {
  std::string label;
  std::chrono::steady_clock::time_point start;
  bool on;

  explicit ScopedTimer(std::string l)
      : label(std::move(l)), on(enabled()) {
    if (on) start = std::chrono::steady_clock::now();
  }
  ~ScopedTimer() {
    if (on) {
      const double s =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
              .count();
      add(label, s);
    }
  }
  ScopedTimer(const ScopedTimer&) = delete;
  ScopedTimer& operator=(const ScopedTimer&) = delete;
};

}  // namespace delta::timing

#define DELTA_TIMER_CONCAT_(a, b) a##b
#define DELTA_TIMER_NAME_(line) DELTA_TIMER_CONCAT_(delta_timer_, line)
#define DELTA_TIME(label) \
  ::delta::timing::ScopedTimer DELTA_TIMER_NAME_(__LINE__) { label }
