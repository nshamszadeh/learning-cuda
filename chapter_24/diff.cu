/*
kernel diff does not have a call to __syncthreads(). This program is used to test Nvidia's racecheck profiling tool.
Code taken from Introduction to High Performance Scientific Computing by DL Chopp
*/
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include "../error_check_cuda.h"

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

// set the threads per block as a constant multiple of the warp size (32 in virtually every case)
const int threadsPerBlock = 256; // 8 full warps per block

// declare the kernel function
__global__ void diff(double* u, int* N, double* dx, double* du);

/**
 * @brief Demonstrate a simple example for implementing a parallel finite difference operator
 * 
 * @param argc Should be 2.
 * @param argv[1] Length of the vector of data.
 * @return int.
 */
int main(int argc, char* argv[]) {
    // read in the number of points
    int N = atoi(argv[1]);

    // determine how many blocks are needed for the whole vector
    const int blocksPerGrid = N / threadsPerBlock + (N % threadsPerBlock > 0 ? 1 : 0);

    // allocate host memory
    double* u = (double*)malloc(N * sizeof(double));
    double* du = (double*)malloc(N * sizeof(double));
    double dx = 2 * M_PI / N; // finite difference length

    double* dev_u;
    double* dev_du;
    double* dev_dx;
    int* dev_N;

    // allocate device memory
    CheckError(cudaMalloc((void**)&dev_u, N * sizeof(double)));
    CheckError(cudaMalloc((void**)&dev_du, N * sizeof(double)));
    CheckError(cudaMalloc((void**)&dev_N, sizeof(int)));
    CheckError(cudaMalloc((void**)&dev_dx, sizeof(double)));

    // initialize data on the host
    for (int i = 0; i < N; i++) {
        u[i] = sin(i * dx);
    }

    // copy data to device
    CheckError(cudaMemcpy(dev_u, u, N * sizeof(double), cudaMemcpyHostToDevice));
    CheckError(cudaMemcpy(dev_dx, &dx, sizeof(double), cudaMemcpyHostToDevice));
    CheckError(cudaMemcpy(dev_N, &N, sizeof(int), cudaMemcpyHostToDevice));

    // execute the finite difference kernel
    diff<<<blocksPerGrid, threadsPerBlock>>>(dev_u, dev_N, dev_dx, dev_du);

    // copy the result back to the host
    CheckError(cudaMemcpy(du, dev_du, N * sizeof(double), cudaMemcpyDeviceToHost));

    // clean up allocated device memory
    CheckError(cudaFree(dev_du));
    CheckError(cudaFree(dev_N));
    CheckError(cudaFree(dev_dx));
    CheckError(cudaFree(dev_u));

    for (int i = 0; i < N; i++) {
        printf("%f\n", du[i]);
    }

    // clean up allocated host memory
    free(u);
    free(du);
    return 0;
}


__global__ void diff(double* u, int* N, double* dx, double* du) {
    // shared memory is implicitly declared static
    __shared__ double local_u[threadsPerBlock + 2]; // need 2 more spaces for computing the finite difference of boundary points
    __shared__ double local_du[threadsPerBlock];

    // set up global and shared memory indices
    int g_i = (threadIdx.x + blockIdx.x * blockDim.x) % *N;
    int l_i = threadIdx.x + 1; // add 1 to index because local_u[0] should be one less (mod N) than the first index of the local data
    int g_im = (g_i + *N - 1) % *N; // we add N and subtract 1 as opposed to just subtract 1 because data is periodic and we dont want a negative index
    int g_ip = (g_i + 1) % *N;

    // Transfer global memory to shared memory
    local_u[l_i] = u[g_i];
    if (threadIdx.x == 0) { // if this thread corresponds to the first element in our local data
        local_u[0] = u[g_im]; // make sure to set the 0th index of local_u to one minus the first index + N (mod N)
    }
    if (threadIdx.x == threadsPerBlock - 1) { // if this thread corresponds to the last element in our local data
        local_u[l_i + 1] = u[g_ip]; // set the last element of local_u 
    }
    __syncthreads();
    // compute the central finite difference for this thread and store in shared memory
    local_du[threadIdx.x] = (local_u[threadIdx.x + 2] - local_u[threadIdx.x]) / (*dx) / 2;
    
    // transfer the results from shared memory to global memory
    du[g_i] = local_du[threadIdx.x];
}