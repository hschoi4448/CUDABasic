
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <stdio.h>
#include <chrono>

using namespace cv;
using namespace std;
using namespace std::chrono;

__global__ void boxBlurKernel(uchar* _dst, const uchar* _src, int rows, int cols, int _n)
{
    // get thread idx from built-in variables
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    // boundary check
    if (x > cols - 1 || y > rows - 1) return;

    // set boxblur range
    int startX = x - _n;
    int startY = y - _n;
    int endX = x + _n;
    int endY = y + _n;

    if (startX < 0) startX = 0;
    if (startY < 0) startY = 0;
    if (endX >= cols - 1) endX = cols - 1;
    if (endY >= rows - 1) endY = rows - 1;

    // calculate average
    float sum = 0;
    int cnt = 0;
    for (int i = startY; i <= endY; i++) {
        for (int j = startX; j <= endX; j++) {
            int idx = i * cols + j;
            sum += _src[idx];
            cnt++;
        }
    }
    float avg = sum / cnt;

    // write result
    _dst[y * cols + x] = (uchar)avg;
}

void boxBlurGPU(Mat* _dst, Mat* _src, int _n)
{
    // device memory pointers
    uchar* d_dst;
    uchar* d_src;

    int cols = _src->cols;
    int rows = _src->rows;
    int dSize = cols * rows * sizeof(uchar);

    // allocate device memory
    cudaMalloc((void**)& d_src, dSize);
    cudaMalloc((void**)& d_dst, dSize);

    // copy src to device memroy
    cudaMemcpy(d_src, _src->data, dSize, cudaMemcpyHostToDevice);

    // prepare boxblur kernel call
    dim3 threads(16, 16);
    dim3 blocks((cols + 15) / 16, (rows + 15) / 16);

    // boxblur kernel call
    boxBlurKernel << <blocks, threads >> > (d_dst, d_src, rows, cols, _n);

    // copy blurred image from device memroy
    cudaMemcpy(_dst->data, d_dst, dSize, cudaMemcpyDeviceToHost);

    // release device memory
    cudaFree(d_dst);
    cudaFree(d_src);
}

void boxBlurCPU(Mat* _dst, Mat* _src, int _n)
{
    // device memory pointers
    uchar* h_dst;
    uchar* h_src;

    int cols = _src->cols;
    int rows = _src->rows;
    int dSize = cols * rows * sizeof(uchar);

    // allocate host memory
    h_src = (uchar*)malloc(dSize);
    h_dst = (uchar*)malloc(dSize);

    // copy src memory
    memcpy(h_src, _src->data, dSize);

    // boxBlur
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            // set boxblur range
            int startX = x - _n;
            int startY = y - _n;
            int endX = x + _n;
            int endY = y + _n;

            if (startX < 0) startX = 0;
            if (startY < 0) startY = 0;
            if (endX >= cols - 1) endX = cols - 1;
            if (endY >= rows - 1) endY = rows - 1;

            // calculate average
            float sum = 0;
            int cnt = 0;
            for (int i = startY; i <= endY; i++) {
                for (int j = startX; j <= endX; j++) {
                    int idx = i * cols + j;
                    sum += h_src[idx];
                    cnt++;
                }
            }
            float avg = sum / cnt;

            // write result
            h_dst[y * cols + x] = (uchar)avg;
        }
    }

    // copy blurred image 
    memcpy(_dst->data, h_dst, dSize);

    // release device memory
    free(h_dst);
    free(h_src);
}

int main()
{
    cout << "number '1','2' change filter type\n";
    cout << "number '3','4' change filter size\n";

    // Read image
    Mat src = imread("../../data/Lenna.png", IMREAD_COLOR);
    if (src.empty()) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }
    // convert to gray 
    cvtColor(src, src, COLOR_BGR2GRAY);

    // create dst mat
    Mat dst = src.clone();

    imshow("Original", src);

    int blurSize = 0;
    int maxBlurSize = 20;
    int fType = 0;
    int fNum = 2;

    // one CUDA boxblur execution for CUDA initialize context
    boxBlurGPU(&dst, &src, 0);

    while (true)
    {
        milliseconds start = duration_cast<milliseconds>(system_clock::now().time_since_epoch());

        if (fType == 0) {
            cout << "boxBlurCPU size: " << blurSize << " ";
            boxBlurCPU(&dst, &src, blurSize);
        }
        if (fType == 1) {
            cout << "boxBlurGPU size: " << blurSize << " ";
            boxBlurGPU(&dst, &src, blurSize);
        }
        milliseconds end = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
        milliseconds elapsed = end - start;

        cout << "elapsed time: " << elapsed.count() << "ms\n";

        imshow("result", dst);

        int k = waitKey(0);
        // 'esc' to finish
        if (k == 27) {
            break;
        }
        // '1' to increase blur size
        if (k == '0' + 1) {
            fType = (fType - 1 + fNum) % fNum;
        }
        // '2' change filter
        if (k == '0' + 2) {
            fType = (fType + 1) % fNum;
        }
        // '3' to decrease blur size
        if (k == '0' + 3) {
            blurSize -= 1;
            if (blurSize < 0) blurSize = 0;
        }
        // '4' to increase blur size
        if (k == '0' + 4) {
            blurSize += 1;
            if (blurSize > maxBlurSize) blurSize = maxBlurSize;
        }
    }

    return 0;
}
