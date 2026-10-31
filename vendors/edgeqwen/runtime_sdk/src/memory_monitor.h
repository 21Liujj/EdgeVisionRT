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

#pragma once

#include <atomic>
#include <cstddef>
#include <future>

namespace edgeqwen
{
namespace internal
{

class MemoryMonitor
{
public:
    MemoryMonitor();
    ~MemoryMonitor();

    void start();
    void stop();

    size_t getPeakGpuMemory() const;
    size_t getPeakCpuMemory() const;
    size_t getPeakUnifiedMemory() const;
    bool isIntegratedGPU() const;

private:
    void monitor();

    std::atomic_bool mActive{false};
    std::future<void> mTask;
    std::atomic<size_t> mPeakGpuMemory{0};
    size_t mBaselineGpuFreeMemory{0};
    bool mIsIGPU{false};
};

} // namespace internal
} // namespace edgeqwen
