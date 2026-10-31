// M8c: qwen_ask — single-question VLM CLI on libedgeqwen_runtime.a
// Usage: qwen_ask --image crop.jpg --prompt "..." [--max-tokens 64]
// Prints answer text line 1, profile JSON line 2. Exit 0 on success.
#include <edgeqwen/runtime.h>

#include <cstdio>
#include <cstring>
#include <string>

int main(int argc, char** argv) {
  std::string image, prompt = "Describe the cropped object in one sentence. Is it abnormal?";
  int max_tok = 64;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char* f) {
      if (i + 1 >= argc) { std::fprintf(stderr, "missing %s\n", f); return std::string(); }
      return std::string(argv[++i]);
    };
    if (a == "--image") image = need("--image");
    else if (a == "--prompt") prompt = need("--prompt");
    else if (a == "--max-tokens") max_tok = std::stoi(need("--max-tokens"));
    else if (a == "--help" || a == "-h") {
      std::printf("usage: qwen_ask --image IMG --prompt TEXT [--max-tokens N]\n");
      return 0;
    }
    if (image.empty() && a == "--image") return 2;
  }
  if (image.empty()) { std::fprintf(stderr, "need --image\n"); return 2; }
  try {
    edgeqwen::Runtime rt({"/home/nvidia/edgeqwen/engines/llm", "/home/nvidia/edgeqwen/engines/visual"});
    rt.warmup();
    edgeqwen::ChatMessage m;
    m.role = "user";
    m.text = prompt;
    m.imagePaths = {image};
    edgeqwen::GenerationParams p;
    p.maxGenerateLength = max_tok;
    auto r = rt.generate({m}, p);
    std::printf("%s\n", r.text.c_str());
    std::printf("%s\n", rt.getLastProfile().toJson().c_str());
    std::fflush(stdout);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "qwen_ask error: %s\n", e.what());
    return 1;
  }
}
