/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "memory_monitor.h"
#include "common/checkMacros.h"
#include <chrono>
#include <cuda_runtime.h>
#include <sys/resource.h>
#include <thread>

namespace edgeqwen
{
namespace internal
{

namespace
{
std::pair<size_t, size_t> getGpuMemoryInfo()
{
    size_t freeMem, totalMem;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    return {freeMem, totalMem};
}

bool detectIntegratedGPU()
{
    int device{-1};
    CUDA_CHECK(cudaGetDevice(&device));
    int integrated = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, device));
    return integrated == 1;
}
} // namespace

MemoryMonitor::MemoryMonitor()
    : mActive(false)
    , mPeakGpuMemory(0)
    , mBaselineGpuFreeMemory(0)
    , mIsIGPU(false)
{
}

MemoryMonitor::~MemoryMonitor()
{
    stop();
}

void MemoryMonitor::start()
{
    if (mTask.valid())
    {
        mActive = false;
        mTask.get();
    }

    mIsIGPU = detectIntegratedGPU();
    mPeakGpuMemory = 0;

    if (!mIsIGPU)
    {
        auto [freeMem, totalMem] = getGpuMemoryInfo();
        mBaselineGpuFreeMemory = freeMem;
        mActive = true;
        mTask = std::async(std::launch::async, [this]() { monitor(); });
    }
}

void MemoryMonitor::stop()
{
    if (mTask.valid())
    {
        mActive = false;
        mTask.get();
    }
}

size_t MemoryMonitor::getPeakGpuMemory() const
{
    return mPeakGpuMemory;
}

size_t MemoryMonitor::getPeakCpuMemory() const
{
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0)
    {
        return static_cast<size_t>(usage.ru_maxrss) * 1024;
    }
    return 0;
}

size_t MemoryMonitor::getPeakUnifiedMemory() const
{
    return getPeakCpuMemory();
}

bool MemoryMonitor::isIntegratedGPU() const
{
    return mIsIGPU;
}

void MemoryMonitor::monitor()
{
    while (mActive.load())
    {
        auto [currentFreeMem, totalMem] = getGpuMemoryInfo();
        size_t gpuMemoryUsed
            = (currentFreeMem < mBaselineGpuFreeMemory) ? mBaselineGpuFreeMemory - currentFreeMem : 0;
        if (mPeakGpuMemory < gpuMemoryUsed)
        {
            mPeakGpuMemory = gpuMemoryUsed;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

} // namespace internal
} // namespace edgeqwen
