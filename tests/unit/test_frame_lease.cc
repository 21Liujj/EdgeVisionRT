#include <cassert>
#include <cstdio>
#include "edgevision/frame.h"
#include "edgevision/frame_lease.h"

int main() {
  edgevision::FramePool pool(4, 1024 * 1024);

  {
    int s0 = pool.acquire_index();
    assert(s0 >= 0);
    edgevision::Frame& f0 = pool.slot_ref(s0);
    f0.data.assign(100, 'x');
    f0.valid = true;

    {
      edgevision::FrameLease lease(&pool, s0);
      assert(lease.valid());
      assert(lease->data.size() == 100);
      assert(lease.slot() == s0);
    }

    assert(!pool.slot_ref(s0).valid);
  }

  {
    int s1 = pool.acquire_index();
    assert(s1 >= 0);
    edgevision::Frame& f1 = pool.slot_ref(s1);
    f1.data.assign(50, 'y');
    f1.valid = true;

    edgevision::FrameLease a(&pool, s1);
    uint64_t a_epoch = pool.epoch_of(s1);

    edgevision::FrameLease b(std::move(a));
    assert(!a.valid());
    assert(b.valid());
    assert(b->data.size() == 50);

    b.release();
  }

  {
    int sa = pool.acquire_index();
    int sb = pool.acquire_index();
    assert(sa >= 0 && sb >= 0);
    pool.slot_ref(sa).valid = true;
    pool.slot_ref(sb).valid = true;

    edgevision::FrameLease la(&pool, sa);
    edgevision::FrameLease lb(&pool, sb);

    la = std::move(lb);
    assert(!lb.valid());
    assert(la.valid());
    assert(la.slot() == sb);
  }

  {
    int sx = pool.acquire_index();
    assert(sx >= 0);
    pool.slot_ref(sx).valid = true;
    uint64_t e1 = pool.epoch_of(sx);

    edgevision::FrameLease lx(&pool, sx);
    assert(lx.valid());

    pool.release(sx);
    int sy = pool.acquire_index();
    assert(sy == sx);
    pool.slot_ref(sy).valid = true;

    assert(!lx.valid());
  }

  std::printf("[test_frame_lease] PASS\n");
  return 0;
}