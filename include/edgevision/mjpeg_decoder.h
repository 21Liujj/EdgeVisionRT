#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace edgevision {

bool mjpeg_decode_to_bgr(const uint8_t* jpeg_data, std::size_t jpeg_size,
                         std::vector<uint8_t>& bgr_out, int& width, int& height,
                         std::string& error);

}  // namespace edgevision