﻿#pragma once

#include <cuda_runtime.h>
#include <cstdint>

void decode_kernel_invoker(
    float *predict, int NUM_BOX_ELEMENT, int num_bboxes, int num_classes, int ckpt, float confidence_threshold,
    float *invert_affine_matrix, float *parray, int max_objects, cudaStream_t stream);
void nms_kernel_invoker(float *parray, float nms_threshold, int max_objects, cudaStream_t stream, int NUM_BOX_ELEMENT);
