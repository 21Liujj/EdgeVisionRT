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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace edgeqwen
{

struct EngineConfig
{
    std::string engineDir;           //!< LLM engine directory
    std::string multimodalEngineDir; //!< Vision engine directory (empty for text-only models)
};

struct GenerationParams
{
    float temperature = 1.0f;
    float topP = 1.0f;
    int64_t topK = 50;
    int64_t maxGenerateLength = 128;
};

struct ChatMessage
{
    std::string role;                    //!< "user" / "system" / "assistant"
    std::string text;                    //!< Text content
    std::vector<std::string> imagePaths; //!< Image file paths (0 or more)
};

struct GenerationResult
{
    std::string text;                 //!< Generated text
    std::vector<int32_t> tokenIds;    //!< Generated token IDs
    int32_t inputTokenCount = 0;      //!< Prompt token count
    std::string finishReason;         //!< "end_id" / "length" / "stop_words" / "error"
};

struct ProfileSnapshot
{
    double visionMeanMs = 0.0;
    double visionP99Ms = 0.0;
    double prefillMeanMs = 0.0;
    double prefillP99Ms = 0.0;
    double decodeMeanMs = 0.0;
    double decodeP99Ms = 0.0;
    double genTokensPerSecond = 0.0;
    double peakUnifiedMemoryMb = 0.0;

    std::string toJson() const;
};

/*!
 * @brief EdgeQwen C++ Runtime
 *
 * Wraps TensorRT-Edge-LLM's LLMInferenceRuntime in a clean, application-facing API.
 * Supports text + vision inputs (Qwen2-VL and similar multimodal models).
 *
 * @note Not thread-safe for concurrent calls. Multiple threads may share a single
 * Runtime instance, but generate() calls are serialized internally via a mutex —
 * concurrent callers will block and queue, not execute in parallel.
 */
class Runtime
{
public:
    explicit Runtime(EngineConfig const& config);
    ~Runtime();

    Runtime(Runtime const&) = delete;
    Runtime& operator=(Runtime const&) = delete;

    /*!
     * @brief End-to-end single-turn inference
     *
     * If messages contain images, automatically runs vision encoder, then prefill + decode.
     * Returns complete generated text in one call.
     *
     * @param messages Conversation messages (user/system/assistant roles)
     * @param params Generation sampling parameters
     * @return Generated result with text, tokens, and finish reason
     * @throws std::runtime_error if inference fails
     */
    GenerationResult generate(
        std::vector<ChatMessage> const& messages, GenerationParams const& params = {});

    /*!
     * @brief Optional CUDA graph warmup
     *
     * Pre-captures decoding CUDA graphs for stable decode latency.
     * Recommended to call once after construction.
     *
     * @return true on success, false on failure (inference still works without graphs)
     */
    bool warmup();

    ProfileSnapshot getLastProfile() const;
    double getPeakUnifiedMemoryMb() const;

private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace edgeqwen
