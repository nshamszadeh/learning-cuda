/**
 * @file exercise2.cu
 * @author Navid Shamszadeh
 * @brief Constant memory version of the finite difference kernel in exercise1.cu
 * @details Write a constant memory version of the diff kernel where the array dev_u is put in constant memory.
 * Use cudaDeviceGetProperties or cudaDeviceGetAttribue to determine the maximum size of the array dev_u that will
 * fit in the constant memory.
 * @date 2021-04-14
 * 
 */
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include "../error_check_cuda.h"

#ifndef M_PI
#define MPI 3.1415926535897932384626433832795
#endif

// set the threads per block as a constant multiple of the warp size 
const int threadsPerBlock = 256;

// declare constant memory on the device
__constant__ double dev_u[8190]; // total constant memory on rtx 3070 is 65,536 bytes, ie 8192 doubles, save room for other variables
__constant__ int dev_N;
__constant__ double dev_dx;

// declare kernel
__global__ void diff(double* dev_du); // all the other variables are stored in constant memory

/**
 * @brief Demonstrate a simple example for implementing a parallel finite difference operator
 * using shared and constant device memory.
 * @param argc Should be 2.
 * @param argv[1] Length of the vector of data.
 * @return int.
 */
int main(int argc, char* argv[]) {
    int N = atoi(argv[1]);
    if (N > 8190) {
        printf("Error: N (%d) + 2 exceeds maximum constant memory capacity! (65,536 bytes).", N);
        exit(-1);
    }

    const int blocksPerGrid = N / threadsPerBlock + (N % threadsPerBlock > 0 ? 1 : 0);
    
    // allocate host memory
    double* u = (double*)malloc(N * sizeof(double));
    double* du = (double*)malloc(N * sizeof(double));

    // initialize data on the host
    double dx = 2 * M_PI / N;
    for (int i = 0; i < N; i++) {
        u[i] = sin(i * dx);
    }

    // allocate device memory
    double* dev_du;
    CheckError(cudaMalloc((void**)&dev_du, N * sizeof(double)));

    // copy data from  host to device
    // cudaMemcpyToSymbol copies data from host to symbol address (in this case somewhere in constant memory).
    CheckError(cudaMemcpyToSymbol(dev_u, u, N * sizeof(double)));
    CheckError(cudaMemcpyToSymbol(dev_N, &N, sizeof(int)));
    CheckError(cudaMemcpyToSymbol(dev_dx, &dx, sizeof(double)));

    // kernel call no longer needs dev_N, dev_u, or dev_dx to be passed as parameters
    diff<<<blocksPerGrid, threadsPerBlock>>>(dev_du);

    // copy results from device to host
    CheckError(cudaMemcpy(du, dev_du, N * sizeof(double), cudaMemcpyDeviceToHost));

    // notice we don't need to free constant memory pointers
    CheckError(cudaFree(dev_du));
    
    // print the results
    for (int i = 0; i < N; i++) {
        printf("%f\n", du[i]);
    }

    // free host memory
    free(u);
    free(du);
    exit(0);
}

__global__ void diff(double* du) {
    /*
     * TODO: Obtain performance metrics of transferring constant memory to shared memory vs only using constant memory.
     */
    // __shared__ double local_u[threadsPerBlock + 2]; // for now, we don't move constant memory to shared memory
    __shared__ double local_du[threadsPerBlock];

    // define global index
    int g_i = (threadIdx.x + blockIdx.x * blockDim.x) % dev_N;
    
    // since we aren't copying any data from global to shared memory, we do not need to call __syncthreads()
    local_du[threadIdx.x] = (dev_u[(g_i + 1) % dev_N] - dev_u[(g_i + dev_N - 1) % dev_N]) / (2 * dev_dx);

    // copy result into global memory
    du[g_i] = local_du[threadIdx.x];
}
