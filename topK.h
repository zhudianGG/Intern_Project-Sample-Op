#pragma once
#include <cuda_runtime.h>
#include <float.h>
#include <cuda.h>
#include <cudd_fp16.h>
#include "src/utils/tensor.h"
 
template<typename T, int K>
struct topK
{
    T val[K];
    int id[K];

    __device__ void init() {
        for (int i = 0; i < K; i++) {
            id[i] = -1;
            val[i] = FLT_MIN;
        }
    }

    void insertHeap(T data, int data_id) {
        if (id[K - 1] == -1 || val[K - 1] < data) {
            id[K - 1] = data_id;
            val[K - 1] = data;
        }
        // 仅需一轮插入新元素重拍的冒泡排序，其他元素均为有序
        for (int i = K - 2; i >= 0; i--) {
            if (val[i + 1] > val[i]) {
                T tmp = val[i];
                val[i] = val[i + 1];
                val[i + 1] = tmp;
                int tmp_id = id[i];
                id[i] = id[i + 1];
                id[i + 1] = tmp_id;
            }
        }
    }
};

template <typename T>
void launchTopKforBeamSearch(TensorWrapper<T> *probs,
                            TensorWrapper<int> *topk_ids,
                            TensorWrapper<T> *topk_vals,
                            TensorWrapper<int> *final_topk_ids,
                            TensorWrapper<T> *final_topk_vals);

template <typename T>
void launchTopKforBeamSearch(const T* probs,
                            const int batch_size,
                            const int vocab_size,
                            int* topk_ids,
                            T* topk_vals,
                            int* final_topk_ids,
                            T* final_topk_vals);

