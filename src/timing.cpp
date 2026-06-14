#include "delta/timing.hpp"

#include <cstdlib>
#include <mutex>
#include <string>

namespace delta::timing {

namespace {

std::mutex g_mutex;
// First-seen-ordered accumulator. The label count is tiny (a handful per call),
// so a linear scan to accumulate is cheaper than the bookkeeping of a map plus a
// separate order vector.
std::vector<std::pair<std::string, double>> g_stages;

}  // namespace

bool enabled() {
  static const bool e = [] {
    const char* v = std::getenv("DELTA_TIMING");
    return v != nullptr && v[0] != '\0' && std::string(v) != "0";
  }();
  return e;
}

void add(const std::string& label, double seconds) {
  std::lock_guard<std::mutex> lock(g_mutex);
  for (auto& [name, secs] : g_stages) {
    if (name == label) {
      secs += seconds;
      return;
    }
  }
  g_stages.emplace_back(label, seconds);
}

void clear() {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_stages.clear();
}

std::vector<std::pair<std::string, double>> drain() {
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_stages;
}

}  // namespace delta::timing
