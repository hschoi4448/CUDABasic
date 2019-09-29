
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

using namespace cv;
using namespace std;

__global__ void boxBlurKernel(uchar *_dst, const uchar *_src, int rows, int cols, int _n)
{
    // get thread idx from built-in variables
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

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
    for (int i = startY; i <= endY; i ++) {
        for (int j = startX; j <= endX; j++) {
            int idx = i * cols + j;
            sum += _src[idx];
            cnt ++;
        }
    }
    float avg = sum / cnt;

    // write result
    _dst[y * cols + x] = (uchar)avg;
}

void boxBlur(Mat *_dst, Mat *_src, int _n)
{
    // device memory pointers
    uchar *d_dst;
    uchar *d_src;

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
    boxBlurKernel<<<blocks, threads >>>(d_dst, d_src, rows, cols, _n);

    // copy blurred image from device memroy
    cudaMemcpy(_dst->data, d_dst, dSize, cudaMemcpyDeviceToHost);

    // release device memory
    cudaFree(d_dst);
    cudaFree(d_src);
}

__global__ void sobelFilterKernel(uchar* _dst, const uchar* _src, int rows, int cols)
{
    // get thread idx from built-in variables
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    
    // prepare sobel filter
    int startX = x - 1;
    int startY = y - 1;
    int endX = x + 1;
    int endY = y + 1;

    if (startX < 0) startX = 0;
    if (startY < 0) startY = 0;
    if (endX >= cols - 1) endX = cols - 1;
    if (endY >= rows - 1) endY = rows - 1;

    float sx[9] = {-1,  0, +1, -2,  0, +2, -1,  0, +1};
    float sy[9] = {-1, -2, -1,  0,  0,  0, +1, +2, +1};

    // calculate average
    float gx = 0;
    float gy = 0;
    int cnt = 0;
    for (int i = startY; i <= endY; i++) {
        for (int j = startX; j <= endX; j++) {
            int idx = i * cols + j;
            gx += _src[idx] * sx[cnt];
            gy += _src[idx] * sy[cnt];
            cnt++;
        }
    }
    float g = sqrtf(gx * gx + gy * gx);
    if (g < 0) g = 0.0f;
    if (g > 255) g = 255.0f;

    // write result
    _dst[y * cols + x] = (uchar)g;
}

void sobelFilter(Mat* _dst, Mat* _src)
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

    // sobel filter kernel call
    sobelFilterKernel<< <blocks, threads >> > (d_dst, d_src, rows, cols);

    // copy blurred image from device memroy
    cudaMemcpy(_dst->data, d_dst, dSize, cudaMemcpyDeviceToHost);

    // release device memory
    cudaFree(d_dst);
    cudaFree(d_src);
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

    // create dst mat
    Mat dst = src.clone();

    imshow("Original", src);
    
    int blurSize = 10;
    int maxBlurSize = 20;
    int fType = 0;
    int fNum = 2;
    while(true)
    {
        if (fType == 0) boxBlur(&dst, &src, blurSize);    
        if (fType == 1) sobelFilter(&dst, &src);

        imshow("result", dst);

        int k = waitKey(0);
        // 'esc' to finish
        if ( k == 27 ) {
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
