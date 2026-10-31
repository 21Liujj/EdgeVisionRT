// Minimal smoke test for FramePool: acquire/release/epoch semantics.
// No camera, no CUDA: must pass on any host via `ctest`.
#include <cassert>
#include <cstdio>
#include "edgevision/frame.h"

int main() {
  edgevision::FramePool pool(4, 1024);
  assert(pool.slots() == 4);
  assert(pool.busy() == 0);

  edgevision::Frame f;
  int s0 = pool.acquire(f);
  assert(s0 >= 0 && pool.busy() == 1);
  uint64_t e0 = pool.epoch_of(s0);

  int s1 = pool.acquire(f);
  assert(s1 >= 0 && s1 != s0);
  pool.release(s0);
  assert(pool.busy() == 1);

  // Re-acquire must reuse a free slot with bumped epoch (stale detection).
  edgevision::Frame f2;
  int s2 = pool.acquire(f2);
  assert(s2 == s0);
  assert(pool.epoch_of(s2) == e0 + 1);

  // Exhaust the pool: 4 slots, 2 busy -> 2 more OK, 5th fails.
  edgevision::Frame tmp;
  assert(pool.acquire(tmp) >= 0);
  assert(pool.acquire(tmp) >= 0);
  assert(pool.acquire(tmp) == -1);

  std::printf("[test_frame] PASS\n");
  return 0;
}
