/**
Modify example 23.1 to add all the error checking as described in the chapter. 
Try modifying the code to generate errors to verify the error checking is working.
For example, try commenting out line 62 and see what happens.
 */
 #include <stdio.h>
 #include <stdlib.h>
 #include <cuda_runtime_api.h>
 #include "../error_check_cuda.h"
 #include "../cudaDmy.cuh"
 
 #ifndef M_PI
 #define MPI 3.1415926535897932384626433832795
 #endif
 /**
  * @brief void diff(double* u, int* N, double* dx, double* du)
  *        Compute the central difference operator on periodic data
  * @param double* u: Function data, assumed periodic
  * @param int* N: Pointer to the length of the data array
  * @param double* dx: Pointer to the space step size
  * @return doulbe* dx: Pointer to the central difference of the u data
  */
 __global__ void diff(double* u, int* N, double* dx, double* du) {
     // blockIdx is a CUDA provided constant that tells the block index within the grid
     int tid = blockIdx.x;
     // notice there's no loop, each core will perform its operation on 
     // its own entry, but some cores should not participate of they are outside
     // the range
     if (tid < *N) {
         int ip = (tid + 1) % *N;
         int im = (tid + *N - 1) % *N;
         du[tid] = (u[ip] - u[im]) / (*dx) / 2.0;
     }
 }
 
 /**
  * @brief int main(int argc, char* argv[])
  *        Demonstrate a simple example for implementing a parallel finite difference operator
  * @param int argc: Should be 2
  * @param argv[1]: Length of the vector of data
  * @return: returns the initial data and its derivative
  */
 int main(int argc, char* argv[]) {
     int N = atoi(argv[1]); // Get length of vector from input
     // These addresses are in host memory
     double* u = (double*)malloc(N * sizeof(double)); // function data
     double* du = (double*)malloc(N * sizeof(double)); // derivative data
     // These addresses are in device memory
     double* dev_u; // function data
     double* dev_du; // derivative data
     double* dev_dx; // space step size
     int* dev_N; // array length
 
     // allocate memory on the device
     CheckError(cudaMalloc((void**) &dev_u, N * sizeof(double)));
     CheckError(cudaMalloc((void**) &dev_du, N * sizeof(double)));
     CheckError(cudaMalloc((void**) &dev_dx, sizeof(double)));
     CheckError(cudaMalloc((void**) &dev_N, sizeof(int)));
 
     // Initialize the function data on the host
     double dx = 2 * M_PI / N;
     for (int i = 0; i < N; i++) {
         u[i] = sin(i * dx);
     }
 
     // Copy the input data drom the host to the device
     CheckError(cudaMemcpy(dev_dx, &dx, sizeof(double), cudaMemcpyHostToDevice));
     CheckError(cudaMemcpy(dev_u, u, N * sizeof(double), cudaMemcpyHostToDevice));
     CheckError(cudaMemcpy(dev_N, &N, sizeof(int), cudaMemcpyHostToDevice));
 
     // Execute the finite difference kernel using N blocks
     diff<<<N, 1>>>(dev_u, dev_N, dev_dx, dev_du);
 
     // Copy the result from the device back to the host
     CheckError(cudaMemcpy(du, dev_du, N * sizeof(double), cudaMemcpyDeviceToHost));
     for (int i = 0; i < N; i++) {
         printf("%lf\t%lf\n", u[i], du[i]);
     }
     // Clean up all the allocated memory
     CheckError(cudaFree(dev_du));
     CheckError(cudaFree(dev_dx));
     CheckError(cudaFree(dev_N));
     CheckError(cudaFree(dev_u));
     free(u);
     free(du);
     return 0;
 }
 
 