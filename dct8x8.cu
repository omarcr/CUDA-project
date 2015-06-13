
/**
**************************************************************************
* \file dct8x8.cu
* \brief Contains entry point, wrappers to host and device code and benchmark.
*
* This sample implements forward and inverse Discrete Cosine Transform to blocks
* of image pixels (of 8x8 size), as in JPEG standard. The typical work flow is as
* follows:
* 1. Run CPU version (Host code) and measure execution time;
* 2. Run CUDA version (Device code) and measure execution time;
* 3. Output execution timings and calculate CUDA speedup.
*/

#include "Common.h"
#include "DCT8x8_Gold.h"
#include "BmpUtil.h"

/**
*  The number of DCT kernel calls
*/
#define BENCHMARK_SIZE  20

/**
*  The PSNR values over this threshold indicate images equality
*/
#define PSNR_THRESHOLD_EQUAL    40


/**
*  Texture reference that is passed through this global variable into device code.
*  This is done because any conventional passing through argument list way results
*  in compiler internal error. 2008.03.11
*/
texture<float, 2, cudaReadModeElementType> TexSrc;


// includes kernels
#include "dct8x8_kernel1.cuh"
#include "dct8x8_kernel_quantization.cuh"


/**
**************************************************************************
*  Wrapper function for 1st gold version of DCT, quantization and IDCT implementations
*
* \param ImgSrc         [IN] - Source byte image plane
* \param ImgDst         [IN] - Quantized result byte image plane
* \param Stride         [IN] - Stride for both source and result planes
* \param Size           [IN] - Size of both planes
*
* \return Execution time in milliseconds
*/
float WrapperGold1(byte *ImgSrc, byte *ImgDst, int Stride, ROI Size)
{
    //allocate float buffers for DCT and other data
    int StrideF;
    float *ImgF1 = MallocPlaneFloat(Size.width, Size.height, &StrideF);
    float *ImgF2 = MallocPlaneFloat(Size.width, Size.height, &StrideF);

    //convert source image to float representation
    CopyByte2Float(ImgSrc, Stride, ImgF1, StrideF, Size);
    AddFloatPlane(-128.0f, ImgF1, StrideF, Size);

    //create and start CUDA timer
    StopWatchInterface *timerGold = 0;
    sdkCreateTimer(&timerGold);
    sdkResetTimer(&timerGold);

    //perform block-wise DCT processing and benchmarking
    for (int i=0; i<BENCHMARK_SIZE; i++)
    {
        sdkStartTimer(&timerGold);
        computeDCT8x8Gold1(ImgF1, ImgF2, StrideF, Size);
        sdkStopTimer(&timerGold);
    }

    //stop and destroy CUDA timer
    float TimerGoldSpan = sdkGetAverageTimerValue(&timerGold);
    sdkDeleteTimer(&timerGold);

    //perform quantization
    quantizeGoldFloat(ImgF2, StrideF, Size);

    //perform block-wise IDCT processing
    computeIDCT8x8Gold1(ImgF2, ImgF1, StrideF, Size);

    //convert image back to byte representation
    AddFloatPlane(128.0f, ImgF1, StrideF, Size);
    CopyFloat2Byte(ImgF1, StrideF, ImgDst, Stride, Size);

    //free float buffers
    FreePlane(ImgF1);
    FreePlane(ImgF2);

    //return time taken by the operation
    return TimerGoldSpan;
}


/**
**************************************************************************
*  Wrapper function for 1st CUDA version of DCT, quantization and IDCT implementations
*
* \param ImgSrc         [IN] - Source byte image plane
* \param ImgDst         [IN] - Quantized result byte image plane
* \param Stride         [IN] - Stride for both source and result planes
* \param Size           [IN] - Size of both planes
*
* \return Execution time in milliseconds
*/
float WrapperCUDA1(byte *ImgSrc, byte *ImgDst, int Stride, ROI Size)
{
    //prepare channel format descriptor for passing texture into kernels
    cudaChannelFormatDesc floattex = cudaCreateChannelDesc<float>();

    //allocate device memory
    cudaArray *Src;
    float *Dst;
    size_t DstStride;
    checkCudaErrors(cudaMallocArray(&Src, &floattex, Size.width, Size.height));
    checkCudaErrors(cudaMallocPitch((void **)(&Dst), &DstStride, Size.width * sizeof(float), Size.height));
    DstStride /= sizeof(float);

    //convert source image to float representation
    int ImgSrcFStride;
    float *ImgSrcF = MallocPlaneFloat(Size.width, Size.height, &ImgSrcFStride);
    CopyByte2Float(ImgSrc, Stride, ImgSrcF, ImgSrcFStride, Size);
    AddFloatPlane(-128.0f, ImgSrcF, ImgSrcFStride, Size);

    //copy from host memory to device
    checkCudaErrors(cudaMemcpy2DToArray(Src, 0, 0,
                                        ImgSrcF, ImgSrcFStride * sizeof(float),
                                        Size.width * sizeof(float), Size.height,
                                        cudaMemcpyHostToDevice));

    //setup execution parameters
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(Size.width / BLOCK_SIZE, Size.height / BLOCK_SIZE);

    //create and start CUDA timer
    StopWatchInterface *timerCUDA = 0;
    sdkCreateTimer(&timerCUDA);
    sdkResetTimer(&timerCUDA);

    //execute DCT kernel and benchmark
    checkCudaErrors(cudaBindTextureToArray(TexSrc, Src));

    for (int i=0; i<BENCHMARK_SIZE; i++)
    {
        sdkStartTimer(&timerCUDA);
        CUDAkernel1DCT<<< grid, threads >>>(Dst, (int) DstStride, 0, 0);
        checkCudaErrors(cudaDeviceSynchronize());
        sdkStopTimer(&timerCUDA);
    }

    checkCudaErrors(cudaUnbindTexture(TexSrc));
    getLastCudaError("Kernel execution failed");

    // finalize CUDA timer
    float TimerCUDASpan = sdkGetAverageTimerValue(&timerCUDA);
    sdkDeleteTimer(&timerCUDA);

    // execute Quantization kernel
    CUDAkernelQuantizationFloat<<< grid, threads >>>(Dst, (int) DstStride);
    getLastCudaError("Kernel execution failed");

    //copy quantized coefficients from host memory to device array
    checkCudaErrors(cudaMemcpy2DToArray(Src, 0, 0,
                                        Dst, DstStride *sizeof(float),
                                        Size.width *sizeof(float), Size.height,
                                        cudaMemcpyDeviceToDevice));

    // execute IDCT kernel
    checkCudaErrors(cudaBindTextureToArray(TexSrc, Src));
    CUDAkernel1IDCT<<< grid, threads >>>(Dst, (int) DstStride, 0, 0);
    checkCudaErrors(cudaUnbindTexture(TexSrc));
    getLastCudaError("Kernel execution failed");

    //copy quantized image block to host
    checkCudaErrors(cudaMemcpy2D(ImgSrcF, ImgSrcFStride *sizeof(float),
                                 Dst, DstStride *sizeof(float),
                                 Size.width *sizeof(float), Size.height,
                                 cudaMemcpyDeviceToHost));

    //convert image back to byte representation
    AddFloatPlane(128.0f, ImgSrcF, ImgSrcFStride, Size);
    CopyFloat2Byte(ImgSrcF, ImgSrcFStride, ImgDst, Stride, Size);

    //clean up memory
    checkCudaErrors(cudaFreeArray(Src));
    checkCudaErrors(cudaFree(Dst));
    FreePlane(ImgSrcF);

    //return time taken by the operation
    return TimerCUDASpan;
}




/**
**************************************************************************
*  Program entry point
*
* \param argc       [IN] - Number of command-line arguments
* \param argv       [IN] - Array of command-line arguments
*
* \return Status code
*/


int main(int argc, char **argv)
{
    //
    // Sample initialization
    //
    printf("%s Starting...\n\n", argv[0]);

    //initialize CUDA
    findCudaDevice(argc, (const char **)argv);

    //source and results image filenames
    char SampleImageFname[] = "barbara.bmp";
    char SampleImageFnameResGold1[] = "barbara_gold1.bmp";
    char SampleImageFnameResCUDA1[] = "barbara_cuda1.bmp";

    char *pSampleImageFpath = sdkFindFilePath(SampleImageFname, argv[0]);

    if (pSampleImageFpath == NULL)
    {
        printf("dct8x8 could not locate Sample Image <%s>\nExiting...\n", pSampleImageFpath);
        exit(EXIT_FAILURE);
    }

    //preload image (acquire dimensions)
    int ImgWidth, ImgHeight;
    ROI ImgSize;
    int res = PreLoadBmp(pSampleImageFpath, &ImgWidth, &ImgHeight);
    ImgSize.width = ImgWidth;
    ImgSize.height = ImgHeight;

    //CONSOLE INFORMATION: saying hello to user
    printf("CUDA sample DCT/IDCT implementation\n");
    printf("===================================\n");
    printf("Loading test image: %s... ", SampleImageFname);

    if (res)
    {
        printf("\nError: Image file not found or invalid!\n");
        exit(EXIT_FAILURE);
        return 1;
    }

    //check image dimensions are multiples of BLOCK_SIZE
    if (ImgWidth % BLOCK_SIZE != 0 || ImgHeight % BLOCK_SIZE != 0)
    {
        printf("\nError: Input image dimensions must be multiples of 8!\n");
        exit(EXIT_FAILURE);
        return 1;
    }

    printf("[%d x %d]... ", ImgWidth, ImgHeight);

    //allocate image buffers
    int ImgStride;
    byte *ImgSrc = MallocPlaneByte(ImgWidth, ImgHeight, &ImgStride);
    byte *ImgDstGold1 = MallocPlaneByte(ImgWidth, ImgHeight, &ImgStride);
    byte *ImgDstCUDA1 = MallocPlaneByte(ImgWidth, ImgHeight, &ImgStride);
  

    //load sample image
    LoadBmpAsGray(pSampleImageFpath, ImgStride, ImgSize, ImgSrc);

    //
    // RUNNING WRAPPERS
    //

    //compute Gold 1 version of DCT/quantization/IDCT
    printf("Success\nRunning Gold 1 (CPU) version... ");
    float TimeGold1 = WrapperGold1(ImgSrc, ImgDstGold1, ImgStride, ImgSize);

    //compute CUDA 1 version of DCT/quantization/IDCT
    printf("Success\nRunning CUDA 1 (GPU) version... ");
    float TimeCUDA1 = WrapperCUDA1(ImgSrc, ImgDstCUDA1, ImgStride, ImgSize);

    //
    // Execution statistics, result saving and validation
    //

    //dump result of Gold 1 processing
    printf("Success\nDumping result to %s... ", SampleImageFnameResGold1);
    DumpBmpAsGray(SampleImageFnameResGold1, ImgDstGold1, ImgStride, ImgSize);


    //dump result of CUDA 1 processing
    printf("Success\nDumping result to %s... ", SampleImageFnameResCUDA1);
    DumpBmpAsGray(SampleImageFnameResCUDA1, ImgDstCUDA1, ImgStride, ImgSize);


	  //print speed info
    printf("Success\n");

    printf("Processing time (CUDA 1)    : %f ms \n", TimeCUDA1);
  
    //calculate PSNR between each pair of images
    float PSNR_Src_DstGold1      = CalculatePSNR(ImgSrc, ImgDstGold1, ImgStride, ImgSize);
    float PSNR_Src_DstCUDA1      = CalculatePSNR(ImgSrc, ImgDstCUDA1, ImgStride, ImgSize);
    float PSNR_DstGold1_DstCUDA1 = CalculatePSNR(ImgDstGold1, ImgDstCUDA1, ImgStride, ImgSize);

    printf("PSNR Original    <---> CPU(Gold 1)    : %f\n", PSNR_Src_DstGold1);
    printf("PSNR Original    <---> GPU(CUDA 1)    : %f\n", PSNR_Src_DstCUDA1);
    printf("PSNR CPU(Gold 1) <---> GPU(CUDA 1)    : %f\n", PSNR_DstGold1_DstCUDA1);

    bool bTestResult = (PSNR_DstGold1_DstCUDA1 > PSNR_THRESHOLD_EQUAL);

    //
    // Finalization
    //

    //release byte planes
    FreePlane(ImgSrc);
    FreePlane(ImgDstGold1);
    FreePlane(ImgDstCUDA1);

    //finalize
    printf("\nTest Summary...\n");
    cudaDeviceReset();

    if (!bTestResult)
    {
        printf("Test failed!\n");
        exit(EXIT_FAILURE);
    }

    printf("Test passed\n");
    exit(EXIT_SUCCESS);
}