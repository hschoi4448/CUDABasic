
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

__global__ void rotateGPUKernel(uchar *_dst, cudaTextureObject_t tex, int _rows, int _cols, float _theta)
{
    // get thread idx from built-in variables
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    // boundary check
    if (x > _cols - 1 || y > _rows - 1) return;

    float u = (float)x - (float)_cols / 2;
    float v = (float)y - (float)_rows / 2;
    float tu = u * cosf(_theta) - v * sinf(_theta);
    float tv = v * cosf(_theta) + u * sinf(_theta);

    tu /= (float)_cols;
    tv /= (float)_rows;

    // read from texture and write to global memory
    _dst[y * _cols + x] = tex2D<float>(tex, tu + 0.5f, tv + 0.5f) * 255;
}

void rotateGPU(Mat *_dst, Mat *_src, int _angle)
{
    // device memory pointers
    uchar *d_dst;

    int cols = _src->cols;
    int rows = _src->rows;
    int dSize = cols * rows * sizeof(uchar);

    cudaMalloc((void **)& d_dst, dSize);

    // allocate cudaArray for texture and copy image to device
    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
    cudaArray *cuArray;
    cudaMallocArray(&cuArray, &channelDesc, cols, rows);
    cudaMemcpyToArray(cuArray, 0, 0, _src->data, dSize, cudaMemcpyHostToDevice);

    // create texture object
    cudaTextureObject_t         tex;
    cudaResourceDesc            texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));

    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = cuArray;
    
    cudaTextureDesc             texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = true;
    texDescr.filterMode = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeWrap;
    texDescr.addressMode[1] = cudaAddressModeWrap;
    texDescr.readMode = cudaReadModeNormalizedFloat;

    cudaCreateTextureObject(&tex, &texRes, &texDescr, NULL);

    // prepare kernel call
    dim3 threads(16, 16);
    dim3 blocks((cols + 15) / 16, (rows + 15) / 16);

    // kernel call
    float rangle = float(_angle) * CV_PI / 180.0f;
    rotateGPUKernel << <blocks, threads >> > (d_dst, tex, rows, cols, rangle);

    // copy blurred image from device memroy
    cudaMemcpy(_dst->data, d_dst, dSize, cudaMemcpyDeviceToHost);

// release device memory
cudaFree(d_dst);
cudaFreeArray(cuArray);
cudaDestroyTextureObject(tex);
}

int main()
{
    cout << "number '1','2' change angle\n";

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

    int theta = 45;

    // warm up
    rotateGPU(&dst, &src, theta);

    while (true)
    {
        cout << "angle: " << theta << "\n";

        rotateGPU(&dst, &src, theta);

        imshow("result", dst);

        int k = waitKey(0);
        // 'esc' to finish
        if (k == 27) {
            break;
        }
        // '1' to increase blur size
        if (k == '0' + 1) {
            theta = (theta + 359) % 360;
        }
        // '2' change filter
        if (k == '0' + 2) {
            theta = (theta + 1) % 360;
        }
    }

    return 0;
}
