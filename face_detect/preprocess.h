#pragma once

#include <cuda_runtime.h>
#include <cstdint>

void preprocess_kernel_img(uint8_t *src, int src_width, int src_height,
                           float *dst, int dst_width, int dst_height,
                           float *d2i, cudaStream_t stream);
