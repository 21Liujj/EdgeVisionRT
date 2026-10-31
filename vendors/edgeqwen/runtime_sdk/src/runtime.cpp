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

#include "edgeqwen/runtime.h"
#include "common/checkMacros.h"
#include "common/trtUtils.h"
#include "memory_monitor.h"
#include "profile_report.h"
#include "profiling/metrics.h"
#include "profiling/timer.h"
#include "runtime/imageUtils.h"
#include "runtime/llmInferenceRuntime.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/streaming.h"
#include <cuda_runtime.h>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace edgeqwen
{

namespace
{

class CudaStreamWrapper
{
public:
    CudaStreamWrapper()
    {
        CUDA_CHECK(cudaStreamCreateWithFlags(&mStream, cudaStreamNonBlocking));
    }

    ~CudaStreamWrapper()
    {
        if (mStream != nullptr)
        {
            cudaStreamDestroy(mStream);
        }
    }

    CudaStreamWrapper(CudaStreamWrapper const&) = delete;
    CudaStreamWrapper& operator=(CudaStreamWrapper const&) = delete;

    CudaStreamWrapper(CudaStreamWrapper&& other) noexcept
        : mStream(other.mStream)
    {
        other.mStream = nullptr;
    }

    CudaStreamWrapper& operator=(CudaStreamWrapper&& other) noexcept
    {
        if (this != &other)
        {
            if (mStream != nullptr)
            {
                cudaStreamDestroy(mStream);
            }
            mStream = other.mStream;
            other.mStream = nullptr;
        }
        return *this;
    }

    cudaStream_t get() const
    {
        return mStream;
    }

private:
    cudaStream_t mStream{nullptr};
};

trt_edgellm::rt::LLMGenerationRequest::Request convertToRtRequest(
    std::vector<ChatMessage> const& messages)
{
    trt_edgellm::rt::LLMGenerationRequest::Request request;
    std::vector<trt_edgellm::rt::Message> chatMessages;
    std::vector<trt_edgellm::rt::imageUtils::ImageData> imageBuffers;

    for (auto const& msg : messages)
    {
        trt_edgellm::rt::Message chatMsg;
        chatMsg.role = msg.role;

        if (!msg.text.empty())
        {
            trt_edgellm::rt::Message::MessageContent msgContent;
            msgContent.type = "text";
            msgContent.content = msg.text;
            chatMsg.contents.push_back(msgContent);
        }

        for (auto const& imagePath : msg.imagePaths)
        {
            trt_edgellm::rt::Message::MessageContent msgContent;
            msgContent.type = "image";
            msgContent.content = imagePath;
            chatMsg.contents.push_back(msgContent);

            auto image = trt_edgellm::rt::imageUtils::loadImageFromFile(imagePath);
            imageBuffers.push_back(std::move(image));
        }

        chatMessages.push_back(chatMsg);
    }

    request.messages = std::move(chatMessages);
    request.imageBuffers = std::move(imageBuffers);
    return request;
}

} // namespace

struct Runtime::Impl
{
    CudaStreamWrapper mStream;
    std::unique_ptr<void, trt_edgellm::DlDeleter> mPluginHandle;
    std::unique_ptr<trt_edgellm::rt::LLMInferenceRuntime> mRuntime;
    std::mutex mCallMutex;
    internal::MemoryMonitor mMemoryMonitor;
    ProfileSnapshot mLastProfile;

    std::vector<float> mVisionTimes;
    std::vector<float> mPrefillTimes;
    std::vector<float> mDecodeTimes;

    Impl(EngineConfig const& config)
    {
        mPluginHandle = trt_edgellm::loadEdgellmPluginLib();
        if (!mPluginHandle)
        {
            throw std::runtime_error("Failed to load EdgeLLM plugin library");
        }

        std::unordered_map<std::string, std::string> loraWeightsMap;
        mRuntime = std::make_unique<trt_edgellm::rt::LLMInferenceRuntime>(
            config.engineDir, config.multimodalEngineDir, loraWeightsMap, mStream.get());

        mMemoryMonitor.start();
    }

    ~Impl()
    {
        mMemoryMonitor.stop();
    }
};

std::string ProfileSnapshot::toJson() const
{
    std::ostringstream oss;
    oss << "{"
        << "\"vision_mean_ms\":" << visionMeanMs << ","
        << "\"vision_p99_ms\":" << visionP99Ms << ","
        << "\"prefill_mean_ms\":" << prefillMeanMs << ","
        << "\"prefill_p99_ms\":" << prefillP99Ms << ","
        << "\"decode_mean_ms\":" << decodeMeanMs << ","
        << "\"decode_p99_ms\":" << decodeP99Ms << ","
        << "\"gen_tokens_per_second\":" << genTokensPerSecond << ","
        << "\"peak_unified_memory_mb\":" << peakUnifiedMemoryMb << "}";
    return oss.str();
}

Runtime::Runtime(EngineConfig const& config)
    : mImpl(std::make_unique<Impl>(config))
{
}

Runtime::~Runtime() = default;

GenerationResult Runtime::generate(std::vector<ChatMessage> const& messages, GenerationParams const& params)
{
    std::lock_guard<std::mutex> lock(mImpl->mCallMutex);

    trt_edgellm::setProfilingEnabled(true);
    trt_edgellm::gTimer.reset();

    trt_edgellm::rt::LLMGenerationRequest llmRequest;
    llmRequest.requests.push_back(convertToRtRequest(messages));
    llmRequest.temperature = params.temperature;
    llmRequest.topP = params.topP;
    llmRequest.topK = params.topK;
    llmRequest.maxGenerateLength = params.maxGenerateLength;

    trt_edgellm::rt::LLMGenerationResponse llmResponse;
    bool success = mImpl->mRuntime->handleRequest(llmRequest, llmResponse, mImpl->mStream.get());

    if (!success)
    {
        throw std::runtime_error("LLM inference failed");
    }

    GenerationResult result;
    if (!llmResponse.outputTexts.empty())
    {
        result.text = llmResponse.outputTexts[0];
    }
    if (!llmResponse.outputIds.empty())
    {
        result.tokenIds = llmResponse.outputIds[0];
    }
    if (!llmResponse.inputTokenCounts.empty())
    {
        result.inputTokenCount = llmResponse.inputTokenCounts[0];
    }
    if (!llmResponse.finishReasons.empty())
    {
        result.finishReason = trt_edgellm::rt::finishReasonName(llmResponse.finishReasons[0]);
    }

    auto visionData = trt_edgellm::gTimer.getTimingData(trt_edgellm::metrics::StageNames::kVISION_ENCODER);
    if (!visionData && !llmRequest.requests[0].imageBuffers.empty())
    {
        visionData = trt_edgellm::gTimer.getTimingData(trt_edgellm::metrics::StageNames::kMULTIMODAL_PROCESSING);
    }
    if (visionData && !visionData->gpuTimesMs.empty())
    {
        mImpl->mVisionTimes.insert(
            mImpl->mVisionTimes.end(), visionData->gpuTimesMs.begin(), visionData->gpuTimesMs.end());
    }

    auto prefillData = trt_edgellm::gTimer.getTimingData(trt_edgellm::metrics::StageNames::kLLM_PREFILL);
    if (prefillData && !prefillData->gpuTimesMs.empty())
    {
        mImpl->mPrefillTimes.insert(
            mImpl->mPrefillTimes.end(), prefillData->gpuTimesMs.begin(), prefillData->gpuTimesMs.end());
    }

    auto decodeData = trt_edgellm::gTimer.getTimingData(trt_edgellm::metrics::StageNames::kLLM_GENERATION);
    if (decodeData && !decodeData->gpuTimesMs.empty())
    {
        mImpl->mDecodeTimes.insert(
            mImpl->mDecodeTimes.end(), decodeData->gpuTimesMs.begin(), decodeData->gpuTimesMs.end());
    }

    auto const& genMetrics = mImpl->mRuntime->getGenerationMetrics();
    double totalDecodeMs = decodeData ? decodeData->getTotalGpuTimeMs() : 0.0;
    double tokensPerSecond = 0.0;
    if (totalDecodeMs > 0.0 && genMetrics.generatedTokens > 0)
    {
        tokensPerSecond = static_cast<double>(genMetrics.generatedTokens) / (totalDecodeMs / 1000.0);
    }

    auto visionStats = internal::StatisticalAnalysis::calculate(mImpl->mVisionTimes);
    auto prefillStats = internal::StatisticalAnalysis::calculate(mImpl->mPrefillTimes);
    auto decodeStats = internal::StatisticalAnalysis::calculate(mImpl->mDecodeTimes);

    mImpl->mLastProfile.visionMeanMs = visionStats.mean;
    mImpl->mLastProfile.visionP99Ms = visionStats.p99;
    mImpl->mLastProfile.prefillMeanMs = prefillStats.mean;
    mImpl->mLastProfile.prefillP99Ms = prefillStats.p99;
    mImpl->mLastProfile.decodeMeanMs = decodeStats.mean;
    mImpl->mLastProfile.decodeP99Ms = decodeStats.p99;
    mImpl->mLastProfile.genTokensPerSecond = tokensPerSecond;
    mImpl->mLastProfile.peakUnifiedMemoryMb = getPeakUnifiedMemoryMb();

    return result;
}

bool Runtime::warmup()
{
    return mImpl->mRuntime->captureDecodingCUDAGraph(mImpl->mStream.get());
}

ProfileSnapshot Runtime::getLastProfile() const
{
    std::lock_guard<std::mutex> lock(mImpl->mCallMutex);
    return mImpl->mLastProfile;
}

double Runtime::getPeakUnifiedMemoryMb() const
{
    if (mImpl->mMemoryMonitor.isIntegratedGPU())
    {
        return mImpl->mMemoryMonitor.getPeakUnifiedMemory() / (1024.0 * 1024.0);
    }
    return mImpl->mMemoryMonitor.getPeakGpuMemory() / (1024.0 * 1024.0);
}

} // namespace edgeqwen
