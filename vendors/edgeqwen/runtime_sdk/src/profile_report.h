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

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace edgeqwen
{
namespace internal
{

struct StatisticalAnalysis
{
    double min{0.0};
    double max{0.0};
    double mean{0.0};
    double median{0.0};
    double p95{0.0};
    double p99{0.0};
    double stddev{0.0};
    size_t count{0};

    static StatisticalAnalysis calculate(std::vector<float> const& data)
    {
        StatisticalAnalysis stats;
        if (data.empty())
        {
            return stats;
        }

        stats.count = data.size();
        stats.mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();

        auto minmax = std::minmax_element(data.begin(), data.end());
        stats.min = static_cast<double>(*minmax.first);
        stats.max = static_cast<double>(*minmax.second);

        double variance = 0.0;
        for (float value : data)
        {
            double dValue = static_cast<double>(value);
            variance += (dValue - stats.mean) * (dValue - stats.mean);
        }
        stats.stddev = std::sqrt(variance / data.size());

        std::vector<float> sorted_data = data;
        std::sort(sorted_data.begin(), sorted_data.end());

        size_t size = sorted_data.size();
        stats.median = static_cast<double>(sorted_data[size / 2]);

        size_t p95_index = std::min(static_cast<size_t>(size * 0.95), size - 1);
        size_t p99_index = std::min(static_cast<size_t>(size * 0.99), size - 1);

        stats.p95 = static_cast<double>(sorted_data[p95_index]);
        stats.p99 = static_cast<double>(sorted_data[p99_index]);

        return stats;
    }
};

} // namespace internal
} // namespace edgeqwen
