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

#include <edgeqwen/runtime.h>
#include <iostream>

int main()
{
    edgeqwen::Runtime rt({"/home/nvidia/edgeqwen/engines/llm", "/home/nvidia/edgeqwen/engines/visual"});

    rt.warmup();

    edgeqwen::ChatMessage message;
    message.role = "user";
    message.text = "Please describe this image.";
    message.imagePaths = {
        "/home/nvidia/edgeqwen/TensorRT-Edge-LLM/examples/multimodal/pics/woman_and_dog.jpeg"};

    auto result = rt.generate({message});

    std::cout << result.text << "\n";
    std::cout << rt.getLastProfile().toJson() << "\n";

    return 0;
}
