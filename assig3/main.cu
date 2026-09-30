#include <iostream>
#include <vector>
#include <curand_kernel.h>

#include "cnpy.h" //https://github.com/rogersce/cnpy


__global__ void init_rng(curandState *states, unsigned long seed, int N){
    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if (idx >= N) return;

    curand_init(seed, idx, 0, &states[idx]);
}


__global__ void init_state(float *x_t, float *y_t, int N){
    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if (idx >= N) return;

    x_t[idx] = 2.5f;
    y_t[idx] = 2.5f;
}


__device__ bool isInBuffer(float x, float y, const float L, const float b){
    bool near_vertical_wall = (x > 0.5f*L - b) && (x < 0.5f*L + b);
    bool near_horizontal_wall = (y > 0.5f*L - b) && (y < 0.5f*L + b);

    return near_vertical_wall || near_horizontal_wall;
}


__global__ void Brownian(float *x_t, float *y_t, int *count, curandState *states, const int N, const int N_threshold, const float L, const float b){
    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if (idx >= N) return;

    curandState local = states[idx];

    float x = x_t[idx];
    float y = y_t[idx];
    int c = count[idx];

    float dx = curand_normal(&local)*0.12f;
    float dy = curand_normal(&local)*0.12f;

    float x_next = x + dx;
    float y_next = y + dy;

    // Always in system
    if (x_next < 0) {x_next *= -1;}
    if (y_next < 0) {y_next *= -1;}
    if (x_next > L) {x_next = 2.0f*L - x_next;}
    if (y_next > L) {y_next = 2.0f*L - y_next;}

    bool crossed = ((x >= 0.5f*L) != (x_next >= 0.5f*L)) || ((y >= 0.5f*L) != (y_next >= 0.5f*L)); // Is particle crossed the wall?

    if (crossed && (c < N_threshold)) {
        // If crossed but c < N_threshold: reject
        x_next = x;
        y_next = y;
    }

    c = isInBuffer(x_next, y_next, L, b) ? c + 1 : 0;

    x_t[idx] = x_next;
    y_t[idx] = y_next;
    count[idx] = c;
    states[idx] = local;
}



int main() {
    const int N = 30; // Number of particles
    const int Nt = 2500; // Number of time stpes
    const int N_threshold = 15; // Number of time stpes holding in the threshold
    const float L = 10.0f; // System size
    const float b = 0.4f; // Buffer size

    int bytes = N*sizeof(float); // Size of float array

    int blockSize = 64; // Threads per block
    int gridSize = (N + blockSize - 1)/blockSize ; // Number of block
    
    // RNG initialization
    curandState *states;
    cudaMalloc(&states, N*sizeof(curandState));
    init_rng<<<gridSize, blockSize>>>(states, 10, N);
    cudaDeviceSynchronize();

    // x Initialization
    float *x, *x_t;
    float *y, *y_t;
    int *count;

    cudaMalloc(&x_t, bytes); // Current x_i(t)
    cudaMalloc(&x, Nt*bytes); // Total trajectories of x_i
    cudaMalloc(&y_t, bytes); // Current y_i(t)
    cudaMalloc(&y, Nt*bytes); // Total trajectories of y_i
    cudaMalloc(&count, N*sizeof(int));

    init_state<<<gridSize, blockSize>>>(x_t, y_t, N);
    cudaDeviceSynchronize();

    // Stochastic dynamics
    for (int t = 0; t < Nt; ++t) {
        cudaMemcpy(x + t*N, x_t, bytes, cudaMemcpyDeviceToDevice);
        cudaMemcpy(y + t*N, y_t, bytes, cudaMemcpyDeviceToDevice);
        Brownian<<<gridSize, blockSize>>>(x_t, y_t, count, states, N, N_threshold, L, b);
    }

    cudaDeviceSynchronize();

    std::vector<float> h_x(Nt*N);
    std::vector<float> h_y(Nt*N);
    std::vector<size_t> shape = {Nt, N};

    cudaMemcpy(h_x.data(), x, Nt*bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y.data(), y, Nt*bytes, cudaMemcpyDeviceToHost);

    cnpy::npz_save("data.npy", "x", h_x.data(), shape, "w");
    cnpy::npz_save("data.npy", "y", h_y.data(), shape, "a");

    cudaFree(x_t);
    cudaFree(x);
    cudaFree(y_t);
    cudaFree(y);
    cudaFree(states);
    cudaFree(count);

    return 0;
}
