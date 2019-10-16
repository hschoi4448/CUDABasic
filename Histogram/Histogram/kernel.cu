
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

// histogram calculation 
void createHistogramCPU(unsigned int *_hist, Mat *_src)
{
    int cols = _src->cols;
    int rows = _src->rows;

    // histogram initialization
    memset(_hist, 0, 256 * sizeof(unsigned int));

    for (int y = 0 ; y < rows; y ++) {
        for (int x = 0; x < cols; x++) {
            int bin = _src->data[y * cols + x];
            _hist[bin]++;
        }
    }
}

// invalid histogram calculation using no atomic operation
__global__ void histogramKernelNoAtomic(unsigned int *_hist, const unsigned char *_src, int _rows, int _cols)
{
    // get thread idx from built-in variables
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    // boundary check
    if (x > _cols - 1 || y > _rows - 1) return;

    int bin = _src[y * _cols + x];
    _hist[bin]++;
}

void createHistogramNoAtomicGPU(unsigned int *_hist, Mat *_src)
{
    int cols = _src->cols;
    int rows = _src->rows;
    int dSize = cols * rows * sizeof(uchar);

    uchar *d_src;
    unsigned int *d_hist;

    // memory allocation
    cudaMalloc((void **)& d_src, dSize);
    cudaMalloc((void **)& d_hist, 256 * sizeof(unsigned int));

    // histogram initialization
    cudaMemset(d_hist, 0, 256 * sizeof(unsigned int));

    // copy image to gpu
    cudaMemcpy(d_src, _src->data, dSize, cudaMemcpyHostToDevice);

    // calculate kernel threads, blocks size
    dim3 threads(16, 16);
    dim3 blocks((cols + 15) / 16, (rows + 15) / 16);

    // kernel call
    histogramKernelNoAtomic << <blocks, threads >> > (d_hist, d_src, rows, cols);

    // copy histogram to host memory
    cudaMemcpy(_hist, d_hist, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // memory release
    cudaFree(d_src);
    cudaFree(d_hist);
}

// histogram calculation with atomic
__global__ void histogramKernel(unsigned int *_hist, const unsigned char *_src, int _rows, int _cols)
{
    // get thread idx from built-in variables
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    // boundary check
    if (x > _cols - 1 || y > _rows - 1) return;

    int bin = _src[y * _cols + x];
    atomicInc(&_hist[bin], UINT_MAX);
}

void createHistogramGPU(unsigned int *_hist, Mat *_src)
{
    int cols = _src->cols;
    int rows = _src->rows;
    int dSize = cols * rows * sizeof(uchar);

    uchar *d_src;
    unsigned int *d_hist;

    // memory allocation
    cudaMalloc((void **)& d_src, dSize);
    cudaMalloc((void **)& d_hist, 256 * sizeof(unsigned int));

    // histogram initialization
    cudaMemset(d_hist, 0, 256 * sizeof(unsigned int));

    // copy image to gpu
    cudaMemcpy(d_src, _src->data, dSize, cudaMemcpyHostToDevice);

    // calculate kernel threads, blocks size
    dim3 threads(16, 16);
    dim3 blocks((cols + 15) / 16, (rows + 15) / 16);

    // kernel call
    histogramKernel << <blocks, threads >> > (d_hist, d_src, rows, cols);

    // copy histogram to host memory
    cudaMemcpy(_hist, d_hist, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // memory release
    cudaFree(d_src);
    cudaFree(d_hist);
}

int main()
{
    // Read image
    Mat src = imread("../../data/Lenna.png", IMREAD_COLOR);
    if (src.empty()) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }
    // convert to gray 
    cvtColor(src, src, COLOR_BGR2GRAY);

 
    // create histogram variable
    unsigned int histogramCPU[256];
    unsigned int histogramGPU[256];
    unsigned int histogramInvalidGPU[256];

    createHistogramCPU(histogramCPU, &src);
    createHistogramGPU(histogramGPU, &src);
    createHistogramNoAtomicGPU(histogramInvalidGPU, &src);


	int eidx = 0;
    // compare histogram from CPU and GPU
    bool flag = true;
    for (int idx = 0; idx < 256; idx++) {
        if (histogramCPU[idx] != histogramGPU[idx]) {
			eidx = idx;
            flag = false;
            break;
        }
    }
    if (flag == true) {
        cout << "CPU and GPU histogram is same\n";
    }
    else {
        cout << "CPU and GPU histogram is different\n";
		cout << "idx:" << eidx << " CPU hist:" << histogramCPU[eidx] << " GPU hist:" << histogramInvalidGPU[eidx] << "\n";
    }

	cout << "\n\n";
    // compare histogram from CPU and histogram from GPU with no atomic operation
    flag = true;
    for (int idx = 0; idx < 256; idx++) {
        if (histogramCPU[idx] != histogramInvalidGPU[idx]) {
			eidx = idx;
            flag = false;
            break;
        }
    }
    if (flag == true) {
        cout << "CPU and GPU histogram without atomic is same\n";
    }
    else {
        cout << "CPU and GPU histogram without atomic is different\n";
		cout << "idx:" << eidx << " CPU hist:" << histogramCPU[eidx] << " GPU hist:" << histogramInvalidGPU[eidx] << "\n";
    }

    return 0;
}