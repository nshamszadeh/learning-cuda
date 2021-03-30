#include <stdio.h>
#include <stdlib.h>

/**
 * @brief adds the first two arguments and places the result in the third argument
 * 
 * @param a : addend 
 * @param b : addend
 * @return c : pointer to the address where the result will be stored. Assumed to be in GPU memory
 */
 __global__ void add(int a, int b, int* c) {
   *c = a + b;
 }

/**
 * @brief Demonstration of a simple program that uses a GPU kernel function.
 *        It takes two input values, adds them together on the GPU, and then
 *        brings the results from the GPU back into the CPU for output
 * @param argc : should be 3
 * @param argv[1] : first addend
 * @param argv[2] : second addend 
 */

int main(int argc, char* argv[]) {
  // read the input values
  int a = atoi(argv[1]);
  int b = atoi(argv[2]);

  // c is the storage place for the main memory result
  int c;

  // dev_c is the storage place for the result on the GPU (device)
  int *dev_c;

  // allocate memory on the GPU to store the result
  cudaMalloc((void**)&dev_c, sizeof(int)); // this is honestly ridiculous 

  // use one GPU unit to perform the addition and store the result in dev_c
  add<<<1,1>>>(a, b, dev_c);

  // move the result into main memory from the GPU
  cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);

  printf("%d + %d = %d\n", a, b, c);

  // free the allocated memory on the GPU
  cudaFree(dev_c);

  return 0;
}