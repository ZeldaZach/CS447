#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>

// At the top of your code.
#define gpu_assert(rv) gpu_assert_h((rv), __FILE__, __LINE__)
void
gpu_assert_h(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// CUDA kernel. Each thread takes care of one element of c
__global__ void vecAdd(double *a, double *b, double *c, int n)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;

    if (id == 10) {
    	printf("ID %d\n", id);
    }
 
    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] + b[id];
}
 
int main( int argc, char* argv[] ) {
    // Size of vectors
    int n = 1000;

    // Host input vectors
    double *h_a;
    double *h_b;
    //Host output vector
    double *h_c;

    // Device input vectors
    double *d_a;
    double *d_b;
    //Device output vector
    double *d_c;

    // Size, in bytes, of each vector
    size_t bytes = n * sizeof(double);

    // Allocate memory for each vector on host
    h_a = (double *) malloc(bytes);
    h_b = (double *) malloc(bytes);
    h_c = (double *) malloc(bytes);

    // Allocate memory for each vector on GPU
    printf("A\n");
    cudaMalloc(&d_a, bytes);
    printf("B\n");
    cudaMalloc(&d_b, bytes);
    printf("C\n");
    cudaMalloc(&d_c, bytes);
    printf("D\n");

    int i;
    // Initialize vectors on host
    for (i = 0; i < n; i++) {
        h_a[i] = 1;//sin(i)*sin(i);
        h_b[i] = 2;//cos(i)*cos(i);
    }

    // Copy host vectors to device
    printf("E\n");
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    printf("F\n");
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    printf("G\n");

    int blockSize, gridSize;

    // Number of threads in each thread block
    blockSize = 1024;

    // Number of thread blocks in grid
    gridSize = (int) ceil((float) n / blockSize);

    printf("H\n");

    // Execute the kernel
    cudaDeviceSynchronize();
    //cudaMemcpyAsync();
    vecAdd <<<gridSize, blockSize>>> (d_a, d_b, d_c, n);
    gpu_assert(cudaPeekAtLastError());

    printf("I\n");

    // Copy array back to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    printf("J\n");

    // Sum up vector c and print result divided by n, this should equal
    // 1 within error
    double sum = 0;
    for (i = 0; i < n; i++) {
        //printf("VALUE: %d\n", h_c[i]);
        sum += h_c[i];
    }
    printf("final result: %f\n", sum/(double)n);
 
    // Release device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
 
    // Release host memory
    free(h_a);
    free(h_b);
    free(h_c);
 
    return 0;
}
