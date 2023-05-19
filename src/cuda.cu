#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <stack>
#include "cuda.cuh"
#include "helper.h"

#include <cstring>
#include <cmath>
#define STACK_MAX 128
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
///
/// Algorithm storage
///
// Number of particles in d_particles
unsigned int cuda_particles_count;
// Device pointer to a list of particles
Particle* d_particles;
//Particle* h_particles;
// Device pointer to a histogram of the number of particles contributing to each pixel
unsigned int* d_pixel_contribs;
unsigned int* h_pixel_contribs;
// Device pointer to an index of unique offsets for each pixels contributing colours
unsigned int* d_pixel_index;
unsigned int* cuda_pixel_index;
// Device pointer to storage for each pixels contributing colours
unsigned char* d_pixel_contrib_colours;
unsigned char* h_colours;
// Device pointer to storage for each pixels contributing colours' depth
float* d_pixel_contrib_depth;
float* h_depth;
// The number of contributors d_pixel_contrib_colours and d_pixel_contrib_depth have been allocated for
unsigned int cuda_pixel_contrib_count;
// Host storage of the output image dimensions
int cuda_output_image_width;
int cuda_output_image_height;

unsigned char* cuda_pixel_contrib_colours;
float* cuda_pixel_contrib_depth;

// Device storage of the output image dimensions
__constant__ int D_OUTPUT_IMAGE_WIDTH;
__constant__ int D_OUTPUT_IMAGE_HEIGHT;
__constant__ int D_PARTICLES_COUNT;
//__global__ void cuda_stage1_kernel;
// Pointer to device image data buffer, for storing the output image data, this must be passed to a kernel to be used on device// Pointer to device image data buffer, for storing the output image data, this must be passed to a kernel to be used on device
unsigned char* d_output_image_data;
unsigned char* h_output_image_data;

void cuda_begin(const Particle* init_particles, const unsigned int init_particles_count,
    const unsigned int out_image_width, const unsigned int out_image_height) {
    // These are basic CUDA memory allocations that match the CPU implementation
    // Depending on your optimisation, you may wish to rewrite these (and update cuda_end())

    // Allocate a opy of the initial particles, to be used during computation
    cuda_particles_count = init_particles_count;
    CUDA_CALL(cudaMalloc(&d_particles, init_particles_count * sizeof(Particle)));
    CUDA_CALL(cudaMemcpy(d_particles, init_particles, init_particles_count * sizeof(Particle), cudaMemcpyHostToDevice));

    // Allocate a histogram to track how many particles contribute to each pixel
    CUDA_CALL(cudaMalloc(&d_pixel_contribs, out_image_width * out_image_height * sizeof(unsigned int)));
    // Allocate an index to track where data for each pixel's contributing colour starts/ends
    CUDA_CALL(cudaMalloc(&d_pixel_index, (out_image_width * out_image_height + 1) * sizeof(unsigned int)));
    cuda_pixel_index = (unsigned int*)malloc((out_image_width * out_image_height + 1) * sizeof(unsigned int));
   
    // Init a buffer to store colours contributing to each pixel into (allocated in stage 2)
    d_pixel_contrib_colours = 0;
    // Init a buffer to store depth of colours contributing to each pixel into (allocated in stage 2)
    d_pixel_contrib_depth = 0;
    // This tracks the number of contributes the two above buffers are allocated for, init 0
    cuda_pixel_contrib_count = 0;

    // Allocate output image
    cuda_output_image_width = (int)out_image_width;
    cuda_output_image_height = (int)out_image_height;
    CUDA_CALL(cudaMemcpyToSymbol(D_OUTPUT_IMAGE_WIDTH, &cuda_output_image_width, sizeof(int)));
    CUDA_CALL(cudaMemcpyToSymbol(D_OUTPUT_IMAGE_HEIGHT, &cuda_output_image_height, sizeof(int)));
    CUDA_CALL(cudaMemcpyToSymbol(D_PARTICLES_COUNT, &cuda_particles_count, sizeof(unsigned int)));

    const int CHANNELS = 3;  // RGB
    CUDA_CALL(cudaMalloc(&d_output_image_data, cuda_output_image_width * cuda_output_image_height * CHANNELS * sizeof(unsigned char)));
    h_output_image_data = (unsigned char*)malloc(cuda_output_image_width * cuda_output_image_height * CHANNELS * sizeof(unsigned char));
    h_pixel_contribs = new unsigned int[cuda_output_image_width * cuda_output_image_height];
    //h_particles = new Particle[cuda_particles_count];
}

__global__ void stage1_kernel(Particle const* __restrict__ particles, const unsigned int particles_count,
    unsigned int* contributions, const unsigned int width, const unsigned int height) {

    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < particles_count) {
        const Particle particle = particles[tid];
        int x_min = max(0, (int)roundf(particle.location[0] - particle.radius));
        int y_min = max(0, (int)roundf(particle.location[1] - particle.radius));
        int x_max = min(width - 1, (int)roundf(particle.location[0] + particle.radius));
        int y_max = min(height - 1, (int)roundf(particle.location[1] + particle.radius));

        for (int x = x_min; x <= x_max; ++x) {
            for (int y = y_min; y <= y_max; ++y) {
                const float x_ab = (float)x + 0.5f - particle.location[0];
                const float y_ab = (float)y + 0.5f - particle.location[1];
                const float pixel_distance = sqrtf(x_ab * x_ab + y_ab * y_ab);
                if (pixel_distance <= particle.radius) {
                    const unsigned int pixel_offset = y * width + x;
                    atomicAdd(&contributions[pixel_offset], 1);
                }
            }
        }
    }
}

void cuda_stage1() {
    // Reset the pixel contributions histogram
    cudaMemset(d_pixel_contribs, 0, cuda_output_image_width * cuda_output_image_height * sizeof(unsigned int));

    //// Launch kernel
    const unsigned int block_size = 256;
    const unsigned int grid_size = (cuda_particles_count + block_size - 1) / block_size;
    stage1_kernel <<<grid_size, block_size >>> (d_particles, cuda_particles_count, d_pixel_contribs, cuda_output_image_width, cuda_output_image_height);
    CUDA_CHECK();

//#ifdef VALIDATION
//
//    
//    CUDA_CALL(cudaMemcpy(h_pixel_contribs, d_pixel_contribs, cuda_output_image_width * cuda_output_image_height * sizeof(unsigned int), cudaMemcpyDeviceToHost));
//    CUDA_CALL(cudaMemcpy(h_particles, d_particles, cuda_particles_count * sizeof(Particle), cudaMemcpyDeviceToHost));
//    validate_pixel_contribs(h_particles, cuda_particles_count, h_pixel_contribs, cuda_output_image_width, cuda_output_image_height);
//    //delete[] h_pixel_contribs;
//    //delete[] h_particles;
//
//
//#endif
}

__global__ void stage2a_kernel(Particle const* __restrict__ particles, const unsigned int particles_count,
    unsigned int* contributions, unsigned int const* __restrict__ index, const unsigned int width, const unsigned int height,
    unsigned char* colours, float* depth) {

    //__shared__ int d_pixel_contribs[BLOCK_SIZE];

    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    /*__shared__ unsigned char shared_colours[16*4];
    __shared__ float shared_depth[16];*/
    if (tid < particles_count) {
        const Particle particle = particles[tid];
        const int x_min = max(0, (int)roundf(particle.location[0] - particle.radius));
        const int y_min = max(0, (int)roundf(particle.location[1] - particle.radius));
        const int x_max = min(width - 1, (int)roundf(particle.location[0] + particle.radius));
        const int y_max = min(height - 1, (int)roundf(particle.location[1] + particle.radius));

        for (int x = x_min; x <= x_max; ++x) {
            for (int y = y_min; y <= y_max; ++y) {
                const float x_ab = (float)x + 0.5f - particle.location[0];
                const float y_ab = (float)y + 0.5f - particle.location[1];
                const float pixel_distance = sqrtf(x_ab * x_ab + y_ab * y_ab);
                if (pixel_distance <= particle.radius) {
                    const unsigned int pixel_offset = y * width + x;
                    unsigned int storage_offset = index[pixel_offset] + atomicAdd(&contributions[pixel_offset], 1);

                    colours[4* storage_offset] = particle.color[0];
                    colours[4* storage_offset + 1] = particle.color[1];
                    colours[4* storage_offset + 2] = particle.color[2];
                    colours[4* storage_offset + 3] = particle.color[3];
                    depth[storage_offset] = particle.location[2];

                }
            }
        }
    }
}

__device__ void insertion_sort(float* depth_start, unsigned char* colours_start, const int num_elements) {
    float key;
    unsigned char color[4];
    //int j;
    int l, r, m;
    for (int i = 1; i < num_elements; i++) {
        key = depth_start[i];
        for (int k = 0; k < 4; k++)
            color[k] = colours_start[4 * i + k];
        //j = i - 1;
        l = 0;
        r = i - 1;
 
        while (l <= r) {
            m = (l + r) / 2;
            if (depth_start[m] > key) {
                r = m - 1;
            }
            else {
                l = m + 1;
            }
        }
        for (int j = i - 1; j >= l; j--) {
            depth_start[j + 1] = depth_start[j];
            for (int k = 0; k < 4; k++)
                colours_start[4 * (j + 1) + k] = colours_start[4 * j + k];
        }
        depth_start[l] = key;
        for (int k = 0; k < 4; k++)
            colours_start[4 * l + k] = color[k];
    }
}

__device__ void swap(float& a, float& b) {
    float t = a;
    a = b;
    b = t;
}

__device__ void swap(unsigned char* a, unsigned char* b, int len = 4) {
    for (int i = 0; i < len; ++i) {
        unsigned char t = a[i];
        a[i] = b[i];
        b[i] = t;
    }
}

__device__ int partition(float* depth_start, unsigned char* colours_start, int low, int high, const int length) {
    if (high >= length || low < 0) {
        return low; // Return low if high is out of bounds
    }
    float pivot = depth_start[high];
    int i = (low - 1);
    for (int j = low; j <= high - 1; j++) {
        if (j >= length) {
            break; // Break if j is out of bounds
        }
        if (depth_start[j] < pivot) {
            i++;
            swap(depth_start[i], depth_start[j]);
            swap(colours_start + 4 * i, colours_start + 4 * j);
        }
    }
    swap(depth_start[i + 1], depth_start[high]);
    swap(colours_start + 4 * (i + 1), colours_start + 4 * high);
    return (i + 1);
}

__device__ void quick_sort(float* depth_start, unsigned char* colours_start, int low, int high, const int length) {
    // Create an auxiliary stack
    int stack[STACK_MAX];

    // Initialize top of stack
    int top = -1;

    // Push initial values to the stack
    stack[++top] = low;
    stack[++top] = high;

    // Keep popping elements until stack is not empty
    while (top >= 0) {
        // Pop high and low
        high = stack[top--];
        low = stack[top--];
        if (high >= length || low < 0) {
            printf("????");
            continue;
        }
        int p = partition(depth_start, colours_start, low, high, length);
        if (p - 1 > low) {
            stack[++top] = low;
            stack[++top] = p - 1;
        }
        if (p + 1 < high) {
            stack[++top] = p + 1;
            stack[++top] = high;
        }
    }
}

__global__ void sort_pairs(float* depth_start, unsigned char* colours_start, unsigned int const* __restrict__ pixel_index, int width, int height, bool quicksort=true) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < width && idy < height) {
        int pixel_id = idy * width + idx;
        if (quicksort == true) {
            quick_sort(depth_start + pixel_index[pixel_id], colours_start + 4 * pixel_index[pixel_id], 0, pixel_index[pixel_id + 1] - pixel_index[pixel_id] - 1, pixel_index[pixel_id + 1] - pixel_index[pixel_id]);
        }
        else {
            insertion_sort(depth_start + pixel_index[pixel_id], colours_start + 4 * pixel_index[pixel_id], pixel_index[pixel_id + 1] - pixel_index[pixel_id]);
        }
    }
}


void cuda_stage2() {
    //skip_pixel_index(pixel_contribs, return_pixel_index, out_image_width, out_image_height);
    //skip_sorted_pairs(h_particles, cuda_particles_count, cuda_pixel_index, cuda_output_image_width, cuda_output_image_height, d_pixel_contrib_colours, d_pixel_contrib_depth);
    CUDA_CALL(cudaMemset(d_pixel_index, 0, (cuda_output_image_width * cuda_output_image_height + 1) * sizeof(unsigned int)));

    thrust::device_ptr<unsigned int> dev_ptr_contribs(d_pixel_contribs);
    thrust::device_ptr<unsigned int> dev_ptr_index(d_pixel_index);
    dev_ptr_index[0] = 0;
    thrust::inclusive_scan(dev_ptr_contribs, dev_ptr_contribs + cuda_output_image_width * cuda_output_image_height, dev_ptr_index + 1);
    
    const unsigned int TOTAL_CONTRIBS = dev_ptr_index[cuda_output_image_height * cuda_output_image_width];
    if (TOTAL_CONTRIBS > cuda_pixel_contrib_count) {
        if (d_pixel_contrib_colours) cudaFree (d_pixel_contrib_colours);
        if (d_pixel_contrib_depth) cudaFree (d_pixel_contrib_depth);
        CUDA_CALL(cudaMalloc((void**) & d_pixel_contrib_colours, TOTAL_CONTRIBS * 4 * sizeof(unsigned char)));
        CUDA_CALL(cudaMalloc((void**) & d_pixel_contrib_depth, TOTAL_CONTRIBS * sizeof(float)));
        cuda_pixel_contrib_count = TOTAL_CONTRIBS;
    }
    CUDA_CALL(cudaMemset(d_pixel_contribs, 0, cuda_output_image_height * cuda_output_image_width * sizeof(unsigned int)));
  
    const unsigned int block_size = 256;
    const unsigned int grid_size = (cuda_particles_count + block_size - 1) / block_size;
    stage2a_kernel << <grid_size, block_size >> > (
        d_particles, cuda_particles_count, d_pixel_contribs, d_pixel_index,
        cuda_output_image_width, cuda_output_image_height,
        d_pixel_contrib_colours, d_pixel_contrib_depth);
 
    dim3 block_size_sort(16, 16);
    dim3 grid_size_sort((cuda_output_image_width + block_size_sort.x - 1) / block_size_sort.x,
        (cuda_output_image_height + block_size_sort.y - 1) / block_size_sort.y);

    // toggle *quicksort* to enable quicksort, otherwise will use binary insertion sort.
    sort_pairs << <grid_size_sort, block_size_sort >> > (
        d_pixel_contrib_depth, d_pixel_contrib_colours, d_pixel_index,
        cuda_output_image_width, cuda_output_image_height, true);
    CUDA_CHECK();
//#ifdef VALIDATION
//
//    h_depth = (float*)malloc(TOTAL_CONTRIBS * sizeof(float));
//    h_colours = (unsigned char*)malloc(TOTAL_CONTRIBS * 4 * sizeof(unsigned char));
//    CUDA_CALL(cudaMemcpy(h_depth, d_pixel_contrib_depth, TOTAL_CONTRIBS * sizeof(float), cudaMemcpyDeviceToHost));
//    CUDA_CALL(cudaMemcpy(h_colours, d_pixel_contrib_colours, TOTAL_CONTRIBS * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost));
//    CUDA_CALL(cudaMemcpy(cuda_pixel_index, d_pixel_index, (cuda_output_image_width * cuda_output_image_height + 1) * sizeof(unsigned int), cudaMemcpyDeviceToHost));
//    validate_pixel_index(h_pixel_contribs, cuda_pixel_index, cuda_output_image_width, cuda_output_image_height);
//    validate_sorted_pairs(h_particles, cuda_particles_count, cuda_pixel_index, cuda_output_image_width, cuda_output_image_height, h_colours, h_depth);
//#endif    
}
__global__ void stage3_kernel(
    unsigned char* output_image,
    unsigned char* const __restrict__ pixel_contrib_colours,
    unsigned int* const __restrict__ pixel_index,
    const int width,
    const int height) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < width && idy < height) {
        const int id = idx * width + idy;
        const int index_start = pixel_index[id];
        const int index_end = pixel_index[id + 1];

        for (unsigned int j = index_start; j < index_end; ++j) {
            const float opacity = (float)pixel_contrib_colours[j * 4 + 3] / 255.0f;

            output_image[(id * 3) + 0] = (unsigned char)((float)pixel_contrib_colours[j * 4 + 0] * opacity + (float)output_image[(id * 3) + 0] * (1 - opacity));
            output_image[(id * 3) + 1] = (unsigned char)((float)pixel_contrib_colours[j * 4 + 1] * opacity + (float)output_image[(id * 3) + 1] * (1 - opacity));
            output_image[(id * 3) + 2] = (unsigned char)((float)pixel_contrib_colours[j * 4 + 2] * opacity + (float)output_image[(id * 3) + 2] * (1 - opacity));
        }
    }
}


void cuda_stage3() {
    //skip_blend(d_pixel_index, d_pixel_contrib_colours, return_output_image);
    CUDA_CALL(cudaMemset(d_output_image_data, 255, cuda_output_image_width * cuda_output_image_height * 3 * sizeof(unsigned char)));
    dim3 block_size(16, 16);
    dim3 grid_size((cuda_output_image_width + block_size.x - 1) / block_size.x, (cuda_output_image_height + block_size.y - 1) / block_size.y);
    stage3_kernel << <grid_size, block_size >> > (d_output_image_data, d_pixel_contrib_colours, d_pixel_index, cuda_output_image_width, cuda_output_image_height);

//#ifdef VALIDATION
//     TODO: Uncomment and call the validation function with the correct inputs
//     You will need to copy the data back to host before passing to these functions
//     (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
//    
//    CUDA_CALL(cudaMemcpy(h_output_image_data, d_output_image_data, cuda_output_image_height * cuda_output_image_width * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost));
//    validate_blend(cuda_pixel_index, cuda_pixel_contrib_colours, &h_output_image_data);
//#endif    
}
void cuda_end(CImage* output_image) {
    // This function matches the provided cuda_begin(), you may change it if desired

    // Store return value
    const int CHANNELS = 3;
    output_image->width = cuda_output_image_width;
    output_image->height = cuda_output_image_height;
    output_image->channels = CHANNELS;
    CUDA_CALL(cudaMemcpy(output_image->data, d_output_image_data, cuda_output_image_width * cuda_output_image_height * CHANNELS * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    // Release allocations
    CUDA_CALL(cudaFree(d_pixel_contrib_depth));
    CUDA_CALL(cudaFree(d_pixel_contrib_colours));
    CUDA_CALL(cudaFree(d_output_image_data));
    CUDA_CALL(cudaFree(d_pixel_index));
    CUDA_CALL(cudaFree(d_pixel_contribs));
    CUDA_CALL(cudaFree(d_particles));
    // Return ptrs to nullptr
    d_pixel_contrib_depth = 0;
    d_pixel_contrib_colours = 0;
    d_output_image_data = 0;
    d_pixel_index = 0;
    d_pixel_contribs = 0;
    d_particles = 0;
    delete[] h_pixel_contribs;
    delete[] cuda_pixel_index;
}
void checkCUDAError(const char* msg)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
