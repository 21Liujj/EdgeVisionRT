#include "nanosam_runtime.hpp"
#include "preprocessing.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <cstring>

using namespace nanosam;

static void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n\n"
              << "Options:\n"
              << "  --image PATH           Input image (JPEG/PNG)\n"
              << "  --box x1 y1 x2 y2      Box prompt in original image coordinates\n"
              << "  --encoder PATH         FP16 or INT8 encoder engine\n"
              << "  --decoder PATH         Mask decoder engine\n"
              << "  --output PATH          Output mask PNG (optional)\n"
              << "  --input-tensor PATH    Read preprocessed [1,3,1024,1024] FP32 tensor from binary file instead of --image (optional)\n"
              << "  --help                 Show this message\n\n"
              << "Example (image mode):\n"
              << "  " << prog << " --image img.jpg --box 100 150 300 400 --encoder enc.engine --decoder dec.engine --output mask.png\n\n"
              << "Example (tensor mode, bypasses preprocessing):\n"
              << "  " << prog << " --input-tensor preprocessed.bin --box 100 150 300 400 --encoder enc.engine --decoder dec.engine\n";
}

int main(int argc, char** argv) {
    std::string image_path, tensor_path, encoder_path, decoder_path, output_path;
    Box box = {0, 0, 0, 0};
    bool has_box = false;
    bool help = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            help = true;
        } else if (arg == "--image" && i + 1 < argc) {
            image_path = argv[++i];
        } else if (arg == "--input-tensor" && i + 1 < argc) {
            tensor_path = argv[++i];
        } else if (arg == "--box" && i + 4 < argc) {
            box.x1 = std::stof(argv[++i]);
            box.y1 = std::stof(argv[++i]);
            box.x2 = std::stof(argv[++i]);
            box.y2 = std::stof(argv[++i]);
            has_box = true;
        } else if (arg == "--encoder" && i + 1 < argc) {
            encoder_path = argv[++i];
        } else if (arg == "--decoder" && i + 1 < argc) {
            decoder_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }

    if (help) {
        printUsage(argv[0]);
        return 0;
    }

    if (encoder_path.empty() || decoder_path.empty() || !has_box) {
        std::cerr << "Error: --encoder, --decoder, and --box are required.\n";
        printUsage(argv[0]);
        return 1;
    }
    if (image_path.empty() && tensor_path.empty()) {
        std::cerr << "Error: either --image or --input-tensor is required.\n";
        printUsage(argv[0]);
        return 1;
    }
    if (!image_path.empty() && !tensor_path.empty()) {
        std::cerr << "Error: --image and --input-tensor are mutually exclusive.\n";
        return 1;
    }

    try {
        std::cout << "[NanoSAM C++] Initializing runtime..." << std::endl;
        NanoSamRuntime runtime;

        if (!runtime.loadEncoder(encoder_path)) {
            std::cerr << "Failed to load encoder: " << encoder_path << std::endl;
            return 1;
        }
        std::cout << "[NanoSAM C++] Encoder loaded: " << encoder_path << std::endl;

        if (!runtime.loadDecoder(decoder_path)) {
            std::cerr << "Failed to load decoder: " << decoder_path << std::endl;
            return 1;
        }
        std::cout << "[NanoSAM C++] Decoder loaded: " << decoder_path << std::endl;

        // Preprocessing
        std::vector<float> input_tensor;
        int orig_width = 0, orig_height = 0;

        auto preprocess_t0 = std::chrono::steady_clock::now();
        if (!image_path.empty()) {
            cv::Mat image_bgr = cv::imread(image_path);
            if (image_bgr.empty()) {
                std::cerr << "Failed to read image: " << image_path << std::endl;
                return 1;
            }
            orig_width = image_bgr.cols;
            orig_height = image_bgr.rows;
            input_tensor = preprocessImage(image_path, 1024);
        } else {
            // Tensor mode: load frozen preprocessed tensor.
            input_tensor = loadTensorFromFile(tensor_path);
            if (input_tensor.size() != static_cast<size_t>(1) * 3 * 1024 * 1024) {
                std::cerr << "Error: input tensor size mismatch, expected 1x3x1024x1024 FP32 (" << (1*3*1024*1024*4) << " bytes)" << std::endl;
                return 1;
            }
            // In tensor mode, orig_width/height must be inferred from another source
            // (metadata file) or passed as CLI arg. For simplicity, we require that
            // the user also supplies them when using --input-tensor, OR we default to
            // assuming the preprocessor was configured with a known (H,W).
            //
            // Here we adopt a fallback: read from an adjacent .meta JSON if present,
            // or assume square 1024x1024 original image (placeholder assumption).
            // Production code would parse .meta or require --orig-width/--orig-height.
            std::string meta_path = tensor_path + ".meta";
            std::ifstream meta_f(meta_path);
            if (meta_f.good()) {
                // Minimal JSON parse for {"orig_width": w, "orig_height": h}.
                // For robustness we'd use a proper JSON library; here we do manual search.
                std::string meta_str((std::istreambuf_iterator<char>(meta_f)),
                                      std::istreambuf_iterator<char>());
                // Quick hack: look for "orig_width":123 and "orig_height":456.
                auto find_int = [](const std::string& s, const std::string& key) -> int {
                    size_t pos = s.find("\"" + key + "\"");
                    if (pos == std::string::npos) return -1;
                    pos = s.find(":", pos);
                    if (pos == std::string::npos) return -1;
                    size_t end = s.find_first_not_of(" \t\n", pos + 1);
                    if (end == std::string::npos) return -1;
                    return std::stoi(s.substr(end));
                };
                orig_width = find_int(meta_str, "orig_width");
                orig_height = find_int(meta_str, "orig_height");
                if (orig_width <= 0 || orig_height <= 0) {
                    std::cerr << "Warning: could not parse orig dimensions from " << meta_path << ", defaulting to 1024x1024" << std::endl;
                    orig_width = orig_height = 1024;
                }
            } else {
                std::cerr << "Warning: no .meta file found for tensor input, assuming 1024x1024 original image dimensions" << std::endl;
                orig_width = orig_height = 1024;
            }
        }
        auto preprocess_t1 = std::chrono::steady_clock::now();
        double preprocess_ms = std::chrono::duration<double, std::milli>(preprocess_t1 - preprocess_t0).count();

        std::cout << "[NanoSAM C++] Image size: " << orig_width << "x" << orig_height << std::endl;
        std::cout << "[NanoSAM C++] Box prompt: [" << box.x1 << ", " << box.y1 << ", " << box.x2 << ", " << box.y2 << "]" << std::endl;

        // Encoder
        double enc_latency = 0.0;
        std::vector<float> embeddings = runtime.runEncoder(input_tensor, &enc_latency);
        std::cout << "[NanoSAM C++] Encoder: " << enc_latency << " ms" << std::endl;

        // Decoder
        double dec_latency = 0.0;
        InferenceResult result = runtime.runDecoder(embeddings, box, orig_width, orig_height, &dec_latency);
        std::cout << "[NanoSAM C++] Decoder: " << dec_latency << " ms" << std::endl;
        std::cout << "[NanoSAM C++] IoU predictions (4 slots): [";
        for (size_t i = 0; i < result.iou_predictions.size(); ++i) {
            std::cout << result.iou_predictions[i];
            if (i + 1 < result.iou_predictions.size()) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        // Postprocess
        cv::Mat mask = runtime.postprocessMask(result.low_res_masks, orig_width, orig_height);

        double full_inference = enc_latency + dec_latency;
        std::cout << "[NanoSAM C++] Full inference (enc + dec): " << full_inference << " ms" << std::endl;
        if (!image_path.empty()) {
            std::cout << "[NanoSAM C++] Preprocessing: " << preprocess_ms << " ms (NOT counted in inference time)" << std::endl;
        }

        if (!output_path.empty()) {
            if (!cv::imwrite(output_path, mask)) {
                std::cerr << "Failed to write output mask: " << output_path << std::endl;
                return 1;
            }
            std::cout << "[NanoSAM C++] Mask written: " << output_path << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "[NanoSAM C++] Done." << std::endl;
    return 0;
}
