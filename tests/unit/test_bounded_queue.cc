// Minimal smoke test for BoundedQueue: capacity, drop counting, shutdown.
// No hardware: must pass on any host via `ctest`.
#include <cassert>
#include <cstdio>
#include <string>
#include "edgevision/bounded_queue.h"

int main() {
  edgevision::BoundedQueue<int> q(2);
  assert(q.try_push(1));
  assert(q.try_push(2));
  assert(!q.try_push(3));  // full: drop, never block
  assert(q.dropped() == 1);

  int v = 0;
  assert(q.wait_pop(v, 100) && v == 1);
  assert(q.wait_pop(v, 100) && v == 2);
  assert(!q.wait_pop(v, 50));  // empty: timeout, not hang

  q.shutdown();
  assert(!q.wait_pop(v, 50));  // shutdown wakes waiter with false
  std::printf("[test_bounded_queue] PASS\n");
  return 0;
}
