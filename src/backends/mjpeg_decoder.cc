#include "edgevision/mjpeg_decoder.h"

#include <csetjmp>
#include <cstdio>
#include <string>
extern "C" {
#include <jpeglib.h>
}

namespace edgevision {

namespace {

struct JpegErrorMgr {
  jpeg_error_mgr pub;
  jmp_buf jmp;
};

void jpeg_error_exit(j_common_ptr cinfo) {
  auto* mgr = reinterpret_cast<JpegErrorMgr*>(cinfo->err);
  (*cinfo->err->output_message)(cinfo);
  longjmp(mgr->jmp, 1);
}

}  // namespace

bool mjpeg_decode_to_bgr(const uint8_t* jpeg_data, std::size_t jpeg_size,
                         std::vector<uint8_t>& bgr_out, int& width, int& height,
                         std::string& error) {
  if (!jpeg_data || jpeg_size == 0) {
    error = "mjpeg_decode: null/empty input";
    return false;
  }

  jpeg_decompress_struct cinfo;
  JpegErrorMgr jerr;

  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = jpeg_error_exit;

  if (setjmp(jerr.jmp)) {
    char buf[JMSG_LENGTH_MAX];
    (*cinfo.err->format_message)(reinterpret_cast<j_common_ptr>(&cinfo), buf);
    error = "mjpeg_decode: libjpeg fatal: " + std::string(buf);
    jpeg_destroy_decompress(&cinfo);
    return false;
  }

  jpeg_create_decompress(&cinfo);
  jpeg_mem_src(&cinfo, jpeg_data, jpeg_size);

  if (jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK) {
    error = "mjpeg_decode: bad JPEG header";
    jpeg_destroy_decompress(&cinfo);
    return false;
  }

  cinfo.out_color_space = JCS_EXT_BGR;
  cinfo.scale_num = 1;
  cinfo.scale_denom = 1;

  if (!jpeg_start_decompress(&cinfo)) {
    error = "mjpeg_decode: start_decompress failed";
    jpeg_destroy_decompress(&cinfo);
    return false;
  }

  width = static_cast<int>(cinfo.output_width);
  height = static_cast<int>(cinfo.output_height);
  int row_stride = width * 3;

  std::size_t needed = static_cast<std::size_t>(row_stride) * static_cast<std::size_t>(height);
  bgr_out.resize(needed);

  while (cinfo.output_scanline < cinfo.output_height) {
    uint8_t* row_ptr = bgr_out.data() + static_cast<std::size_t>(cinfo.output_scanline) *
                                             static_cast<std::size_t>(row_stride);
    jpeg_read_scanlines(&cinfo, &row_ptr, 1);
  }

  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  return true;
}

}  // namespace edgevision