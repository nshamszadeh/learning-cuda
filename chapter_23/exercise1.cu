/*
Write a kernel that fills out a two-dimensional array of integers of dimension 100x100.
The kernel should save the value of the function:
    value = 100 * (100 * (100 * blockIdx.x + blockIdx.y) + threadIdx.x) + threadIdx.y;

Experiment with different size numbers of threads per block and verify the values returned are as expected.
*/
//#include "cuda.h"
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include <__clang_cuda_device_functions.h>
#include <__clang_cuda_intrinsics.h>
#include <fstream>
#include <iostream>

__global__ void kernel(long* A, const int N) {
    // assume 10 x 10 blocks of 10 x 10 threads
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    long value = 100 * (100 * (100 * blockIdx.x + blockIdx.y) + threadIdx.x) + threadIdx.y;
    //long value = i + j * N;
    
    if (i < N && j < N) {
        A[i + j * N] = value;
    }
}

int main() {
    const int N = 100;
    // Since this is for a simple exercise there won't be any error checking.
    // Obviously error checking is important for production code.
    long* A = (long*)malloc(N * N * sizeof(long));
    // allocate GPU memory for the array
    long* dev_A;
    cudaMalloc((void**) &dev_A, N * N * sizeof(long)); 
    // specify block and thread layouts
    dim3 threadsPerBlock(25, 25);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    // invoke the kernel
    kernel<<<numBlocks, threadsPerBlock>>>(dev_A, N);
    // copy result into host memory
    cudaMemcpy(A, dev_A, N * N * sizeof(long), cudaMemcpyDeviceToHost);
    // free device memory
    cudaFree(dev_A);
    // write result to a file (writing a 10,000 element array to stdout seems a bit much)
    std::ofstream outfile("result.txt", std::ofstream::out);
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            //outfile << A[col + row * N] << ' ';
            outfile << A[row + col * N] << ' ';
        }
        outfile << std::endl;
    }
    // close output file, free host memory
    outfile.close();
    free(A);
    return 0;
}
