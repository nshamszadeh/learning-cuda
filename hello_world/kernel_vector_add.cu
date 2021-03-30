#include <stdio.h>
#include <chrono>
#include <random>
#include <limits>


void serial_add(float* A, float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}


/**
 * @brief void vecAdd(float* A, float* B, float* C)
 *        Adds elements of vectors A and B and stores them in vector D
 * @param A: vector addend
 * @param B: vector addend
 * @param C: pointer to where the result of A + B will be stored
 * @return C: pointer to address where A + B will be stored
 */
__global__ void vecAdd(float* A, float* B, float* C) {
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main() {
    int N = 1000000000;

    float* A = (float*)malloc(N * sizeof(float));
    float* B = (float*)malloc(N * sizeof(float));
    float* C = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine generator (seed);
        float d = std::generate_canonical<float, std::numeric_limits<float>::digits>(generator);
        A[i] = d;
    }

    for (int i = 0; i < N; i++) {
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine generator (seed);
        float d = std::generate_canonical<float, std::numeric_limits<float>::digits>(generator);
        B[i] = d;
    }

    float* dev_A;
    float* dev_B;
    float* dev_C;

    cudaMalloc((void**) &dev_A, N * sizeof(float));
    cudaMalloc((void**) &dev_B, N * sizeof(float));
    cudaMalloc((void**) &dev_C, N * sizeof(float));

    cudaMemcpy(dev_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, N * sizeof(float), cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();
    vecAdd<<<1, N>>>(dev_A, dev_B, dev_C);
    cudaMemcpy(C, dev_C, N * sizeof(float), cudaMemcpyDeviceToHost);
    auto end = std::chrono::high_resolution_clock::now();
    printf("GPU Compute Time: %u\n", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 25.0);
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);

    for (int i = 0; i < N; i++) {
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine generator (seed);
        float d = std::generate_canonical<float, std::numeric_limits<float>::digits>(generator);
        A[i] = d;
    }

    for (int i = 0; i < N; i++) {
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine generator (seed);
        float d = std::generate_canonical<float, std::numeric_limits<float>::digits>(generator);
        B[i] = d;
    }

    start = std::chrono::high_resolution_clock::now();
    serial_add(A, B, C, N);
    end = std::chrono::high_resolution_clock::now();
    printf("CPU Compute Time: %u\n", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 25.0);

    free(A);
    free(B);
    free(C);

    return 0;
}