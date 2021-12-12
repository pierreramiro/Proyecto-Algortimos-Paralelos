// ING618 - Median Filter
// Sep 2017, Oct 2020, Cesar Carranza

// Updates to Oct 2021:
// NPP libraries has changed in 10.1, now I need to add nppc.lib and nppif.lib in the linker.

// Sep 2017:
// Need: bmw2560Color.tiff and bmw2560Filtered.tiff files in the working directory
// bmw2560Color.tiff is an RGB image, only channel Green is extracted and median filtered.
// Output is stored in bmw2560filtered.tiff (grayscale image, i.e. 1 channel)
//
// 06_GPU: Now using 2D mapping, although... it runs WORSE!
// 05_GPU_sn_shared.cu: Now using shared memory ... ACTUALLY... NEVER DID!, LOL.
//    Kernels are loading the same data so many times, make it now only once....\
//    According to the profiler, Global Mem is accessed only 1 time per pixel (L2 cache miss is only 3.3%!)
//    Cache MISS on L1 is HIGH... 66.7% (k=5), Let's try to improve that.
//    Idea: Launching threads in 1D does not help to data locality... instead we can launch blocks in 2D, something like 16x16, or 32x32
//           So pixels in different image lines also exploit locality. ... done in v06... runs slower :(
//    
// 05_GPU_sn.cu: Version using Sorting Network, only for k=3, 5. Using optimal Batcher.
//               Still can be improved, since we do not need the whole sequence sorted, we need only the median.
//    Also it is optimized:
//    i) Minimum number of PTX instructions using fmin,fmax instead of setp, selp.
//    ii) Memory coalescing: The image HxW is padded to (H+(k-1)/2)*(W+(k-1)/2), however this size is not multiple of 128 bytes
//        Since we use floats, each pixel is 4 bytes, so we need to make sure the size of the image is multiple of 32 pixels (32*8 bytes)
//        So, use as new size Hpad * Wpad =  ((H+(k-1)+1)/32 * 32)) x ((W+(k-1)+1)/32 * 32))
//
// 04_GPU_Naive_bs_clean.cu: Remove all but the GPU MF so I can do profiling
// Try3: Naive version. Use Bubble sort but avoiding the conditional if

// Try2: Naive version. Use k4 sort which does not use conditionals

// Try1: Use the NPP library (03.08 NPP_Library.pdf, page 1321)
// Single channel 32-bit floating-point median filter.
// NppStatus nppiFilterMedian_32f_C1R (const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst,
//                                     Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize,
//                                     NppiPoint oAnchor, Npp8u *pBuffer)
// Parameters:
//  pSrc: Source-Image Pointer.
//  nSrcStep Source-Image Line Step.
//  pDst Destination-Image Pointer.
//  nDstStep Destination-Image Line Step.
//  oSizeROI Region-of-Interest (ROI).
//  oMaskSize Width and Height of the neighborhood region for the local Median operation.
//  oAnchor X and Y offsets of the kernel origin frame of reference relative to the source pixel.
//  pBuffer Pointer to the user-allocated scratch buffer required for the Median operation.
// Returns:
//  Image Data Related Error Codes, ROI Related Error Codes
//
// To compare, 3 Serial algotithms: k4, bs, qs



// Extra info:
//4.3 Region-of-Interest (ROI)
// In practice processing a rectangular sub-region of an image is often more common than processing complete
// images. The vast majority of NPP’s image-processing primitives allow for processing of such sub
// regions also referred to as regions-of-interest or ROIs.
// All primitives supporting ROI processing are marked by a "R" in their name suffix. In most cases the ROI is
// passed as a single NppiSize struct, which provides the with and height of the ROI. This raises the question
// how the primitive knows where in the image this rectangle of (width, height) is located. The "start pixel" of
// the ROI is implicitly given by the image-data pointer. I.e. instead of explicitly passing a pixel coordinate
// for the upper-right corner, the user simply offsets the image-data pointers to point to the first pixel of the
// ROI.
// In practice this means that for an image (pSrc, nSrcStep) and the start-pixel of the ROI being at location
// (xROI, yROI), one would pass
// pSrcOffset = pSrc + yROI * nSrcStep + xROI * PixelSize; WARNING: THIS IS WRONG!!!, pointers are not handled at BYTE level!!!
// as the image-data source to the primitive. PixelSize is typically computed as
// PixelSize = NumberOfColorChannels  sizeof(PixelDataType).
// E.g. for a pimitive like nppiSet_16s_C4R() we would have
// • NumberOfColorChannels == 4;
// • sizeof(Npp16s) == 2;
// • and thus PixelSize = 4 * 2 = 8;


// Issues:
// When using NPP, I had to manually add the libs: I added all.... nppc.lib, nppi.lib, npps.lib
// (Solution Explorer ... RightClick Project Name ...  Properties ... Linker ... Input ... Additional depencies)
// Also, I noticed that those libraries are only available for x64, so make sure your project is for x64

// To generate pxt files
// In Solution Explorer ... RightClick Project Name ...  Properties ...CUDA C/C++ ... Common ... Keep processed files: YES

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <conio.h> // to use getch() as pause
#include <time.h> // to use clock() functions
#include <math.h> // to use abs() function
#include <stdlib.h> // malloc and others
#include <npp.h> // To use the median filter library

/****************************************************************************************/
/* MODIFY THESE PARAMETERS TO SELECT DIFFERENT TYPES OF IMPLEMENTATIONS *****************/
/****************************************************************************************/
// Main selection: what to run. It is mandatory to run at least 1 GPU code.
#define runCPUSerial 1 // 0: None, 1: Only BS, 2: Only QS, 3: QS and BS
#define runGPU 0 // 0: Custom kernel, 1: Run NPP, 2: NPP and Custom kernel
#define pause 1 // To pause the end of the processing (keep the console window open in Win7)

// Image and kernel size. Also padding mode
#define k 3 // Filter size
#define paddingMode 0 // 0: pad to (w+k-1)x(h+k-1), 1: pad to ((w+k-1+31)/32)*32 x ((h+k-1+31)/32)*32
#define imageTest 0 // 0: 2560x2560 BMW, 1: 1: 1920x1080 Audi

// Customization own kernel
#define mode 1 // 1:1D block, 1D grid. 2:2D Block, 1D grid. 3:2D Block, 2D Grid
#define threadsx 32 // threads per block 2D. Make sure the total threads is multiple of this. Remember total threads/block = 1024
#define threadsy 32
#define swapMode 1 // 0: Standard swap with a temporary variable, 1: min/max swap
/****************************************************************************************/
/****************************************************************************************/
/****************************************************************************************/
#if imageTest == 0
#define height 2560 // Image height
#define width 2560	// Image width
#define nameInput "bmw2560Color.tiff"
#define nameOutput "bmw2560Filtered.tiff"
#endif
#if imageTest == 1
#define height 1080 // Image height
#define width 1920	// Image width
#define nameInput "auto1080pColor.tiff"
#define nameOutput "auto1080pFiltered.tiff"
#endif

#if swapMode == 0
#define CMP_GT_SWAP( x, y, t )  (x > y ?  t=x,x=y,y=t : x=x,y=y )	// Compiler fuses all this in 3 PTX instructions
																// setp.gt.f32	%p1, %f1, %f2;   // p1  = (f1 > f2) ?  1 : 0
																// selp.f32	%f10, %f1, %f2, %p1; // f10 = (p1 == 1) ? f1 : f2
																// selp.f32	%f11, %f2, %f1, %p1; // f11 = (p1 == 1) ? f2 : f1
																// Equivalent to: if (x>y) swap(x,y), but no divergence.
#endif
#if swapMode == 1
#define CMP_GT_SWAP( x, y, t ) ( t=fmin(x,y),y=fmax(x,y),x=t ); // Compile to 2 PTX instructions
															// min.f32 	%f10, %f1, %f2; // x_new = min(x,y)
															// max.f32 	%f11, %f1, %f2; // y_new = max(x,y)
															// Again, no divergence, 1 less PTX instruction.
#endif

#if paddingMode == 0
#define widthp width+k-1 // Computed as ((w+k-1+31)/32)*32
#define heightp width+k-1 // Computed as ((h+k-1+31)/32)*32
#endif
#if paddingMode == 1
#define widthp ((width+k-1+31)/32)*32 // Computed as ((w+k-1+31)/32)*32
#define heightp ((height+k-1+31)/32)*32 // Computed as ((h+k-1+31)/32)*32
#endif


void mf_serial_k4(float* filmed, float* ImBase);
void mf_serial_bs(float* filmed, float* ImBase);
void mf_serial_qs(float* filmed, float* ImBase);
void quickSort(float* a, int l, int r);
int partition(float* a, int l, int r);




// CUDA kernel
__global__ void mf_gpu_bs(float* filmedCuda, float* ImBaseCuda)
{
	// Each thread has its own window. WARNING: memory is limited to 64K registers per block
	float buf[k * k], tmp; // Consider using in place sorting, so we use only 1 window.

	unsigned int a, b, x, y; // WARNING: Try to the minimum of variables

#if mode == 1
	y = ((blockIdx.x * blockDim.x) + threadIdx.x) / width; // 1D Block, 1D grid
	x = ((blockIdx.x * blockDim.x) + threadIdx.x) % width;
#endif

#if mode == 2
	x = (blockIdx.x % (width / threadsx)) * threadsx + threadIdx.x; // 2D Blocks, 1D grid
	y = (blockIdx.x / (width / threadsx)) * threadsy + threadIdx.y;
#endif

#if mode == 3
	y = (blockIdx.y * blockDim.y) + threadIdx.y;; // 2D blocks, 2D grid
	x = (blockIdx.x * blockDim.x) + threadIdx.x;
#endif

	//Load analysis area (window)
	for (a = 0; a < k; a++)
	{
		for (b = 0; b < k; b++)
		{
			buf[a * k + b] = ImBaseCuda[(y + a) * (widthp)+x + b];
		}
	}

	// Sorting network: http://pages.ripco.net/~jgamble/nw.html
	//Network for N=9, using Best Known Arrangement.
	//There are 25 comparators in this network,
	//grouped into 9 parallel operations.
	//[[0,1],[3,4],[6,7]]
	//[[1,2],[4,5],[7,8]]
	//[[0,1],[3,4],[6,7],[2,5]]
	//[[0,3],[1,4],[5,8]]
	//[[3,6],[4,7],[2,5]]
	//[[0,3],[1,4],[5,7],[2,6]]
	//[[1,3],[4,6]]
	//[[2,4],[5,6]]
	//[[2,3]]

#if k == 3
	CMP_GT_SWAP(buf[0], buf[1], tmp);
	CMP_GT_SWAP(buf[3], buf[4], tmp);
	CMP_GT_SWAP(buf[6], buf[7], tmp);
	CMP_GT_SWAP(buf[1], buf[2], tmp);
	CMP_GT_SWAP(buf[4], buf[5], tmp);
	CMP_GT_SWAP(buf[7], buf[8], tmp);
	CMP_GT_SWAP(buf[0], buf[1], tmp);
	CMP_GT_SWAP(buf[3], buf[4], tmp);
	CMP_GT_SWAP(buf[6], buf[7], tmp);
	CMP_GT_SWAP(buf[2], buf[5], tmp);
	CMP_GT_SWAP(buf[0], buf[3], tmp);
	CMP_GT_SWAP(buf[1], buf[4], tmp);
	CMP_GT_SWAP(buf[5], buf[8], tmp);
	CMP_GT_SWAP(buf[3], buf[6], tmp);
	CMP_GT_SWAP(buf[4], buf[7], tmp);
	CMP_GT_SWAP(buf[2], buf[5], tmp);
	CMP_GT_SWAP(buf[0], buf[3], tmp);
	CMP_GT_SWAP(buf[1], buf[4], tmp);
	CMP_GT_SWAP(buf[5], buf[7], tmp);
	CMP_GT_SWAP(buf[2], buf[6], tmp);
	CMP_GT_SWAP(buf[1], buf[3], tmp);
	CMP_GT_SWAP(buf[4], buf[6], tmp);
	CMP_GT_SWAP(buf[2], buf[4], tmp);
	CMP_GT_SWAP(buf[5], buf[6], tmp);
	CMP_GT_SWAP(buf[2], buf[3], tmp);
#endif

#if k == 5
	CMP_GT_SWAP(buf[0], buf[16], tmp);
	CMP_GT_SWAP(buf[1], buf[17], tmp);
	CMP_GT_SWAP(buf[2], buf[18], tmp);
	CMP_GT_SWAP(buf[3], buf[19], tmp);
	CMP_GT_SWAP(buf[4], buf[20], tmp);
	CMP_GT_SWAP(buf[5], buf[21], tmp);
	CMP_GT_SWAP(buf[6], buf[22], tmp);
	CMP_GT_SWAP(buf[7], buf[23], tmp);
	CMP_GT_SWAP(buf[8], buf[24], tmp);
	CMP_GT_SWAP(buf[0], buf[8], tmp);
	CMP_GT_SWAP(buf[1], buf[9], tmp);
	CMP_GT_SWAP(buf[2], buf[10], tmp);
	CMP_GT_SWAP(buf[3], buf[11], tmp);
	CMP_GT_SWAP(buf[4], buf[12], tmp);
	CMP_GT_SWAP(buf[5], buf[13], tmp);
	CMP_GT_SWAP(buf[6], buf[14], tmp);
	CMP_GT_SWAP(buf[7], buf[15], tmp);
	CMP_GT_SWAP(buf[16], buf[24], tmp);
	CMP_GT_SWAP(buf[8], buf[16], tmp);
	CMP_GT_SWAP(buf[9], buf[17], tmp);
	CMP_GT_SWAP(buf[10], buf[18], tmp);
	CMP_GT_SWAP(buf[11], buf[19], tmp);
	CMP_GT_SWAP(buf[12], buf[20], tmp);
	CMP_GT_SWAP(buf[13], buf[21], tmp);
	CMP_GT_SWAP(buf[14], buf[22], tmp);
	CMP_GT_SWAP(buf[15], buf[23], tmp);
	CMP_GT_SWAP(buf[0], buf[4], tmp);
	CMP_GT_SWAP(buf[1], buf[5], tmp);
	CMP_GT_SWAP(buf[2], buf[6], tmp);
	CMP_GT_SWAP(buf[3], buf[7], tmp);
	CMP_GT_SWAP(buf[8], buf[12], tmp);
	CMP_GT_SWAP(buf[9], buf[13], tmp);
	CMP_GT_SWAP(buf[10], buf[14], tmp);
	CMP_GT_SWAP(buf[11], buf[15], tmp);
	CMP_GT_SWAP(buf[16], buf[20], tmp);
	CMP_GT_SWAP(buf[17], buf[21], tmp);
	CMP_GT_SWAP(buf[18], buf[22], tmp);
	CMP_GT_SWAP(buf[19], buf[23], tmp);
	CMP_GT_SWAP(buf[4], buf[16], tmp);
	CMP_GT_SWAP(buf[5], buf[17], tmp);
	CMP_GT_SWAP(buf[6], buf[18], tmp);
	CMP_GT_SWAP(buf[7], buf[19], tmp);
	CMP_GT_SWAP(buf[12], buf[24], tmp);
	CMP_GT_SWAP(buf[4], buf[8], tmp);
	CMP_GT_SWAP(buf[5], buf[9], tmp);
	CMP_GT_SWAP(buf[6], buf[10], tmp);
	CMP_GT_SWAP(buf[7], buf[11], tmp);
	CMP_GT_SWAP(buf[12], buf[16], tmp);
	CMP_GT_SWAP(buf[13], buf[17], tmp);
	CMP_GT_SWAP(buf[14], buf[18], tmp);
	CMP_GT_SWAP(buf[15], buf[19], tmp);
	CMP_GT_SWAP(buf[20], buf[24], tmp);
	CMP_GT_SWAP(buf[0], buf[2], tmp);
	CMP_GT_SWAP(buf[1], buf[3], tmp);
	CMP_GT_SWAP(buf[4], buf[6], tmp);
	CMP_GT_SWAP(buf[5], buf[7], tmp);
	CMP_GT_SWAP(buf[8], buf[10], tmp);
	CMP_GT_SWAP(buf[9], buf[11], tmp);
	CMP_GT_SWAP(buf[12], buf[14], tmp);
	CMP_GT_SWAP(buf[13], buf[15], tmp);
	CMP_GT_SWAP(buf[16], buf[18], tmp);
	CMP_GT_SWAP(buf[17], buf[19], tmp);
	CMP_GT_SWAP(buf[20], buf[22], tmp);
	CMP_GT_SWAP(buf[21], buf[23], tmp);
	CMP_GT_SWAP(buf[2], buf[16], tmp);
	CMP_GT_SWAP(buf[3], buf[17], tmp);
	CMP_GT_SWAP(buf[6], buf[20], tmp);
	CMP_GT_SWAP(buf[7], buf[21], tmp);
	CMP_GT_SWAP(buf[10], buf[24], tmp);
	CMP_GT_SWAP(buf[2], buf[8], tmp);
	CMP_GT_SWAP(buf[3], buf[9], tmp);
	CMP_GT_SWAP(buf[6], buf[12], tmp);
	CMP_GT_SWAP(buf[7], buf[13], tmp);
	CMP_GT_SWAP(buf[10], buf[16], tmp);
	CMP_GT_SWAP(buf[11], buf[17], tmp);
	CMP_GT_SWAP(buf[14], buf[20], tmp);
	CMP_GT_SWAP(buf[15], buf[21], tmp);
	CMP_GT_SWAP(buf[18], buf[24], tmp);
	CMP_GT_SWAP(buf[2], buf[4], tmp);
	CMP_GT_SWAP(buf[3], buf[5], tmp);
	CMP_GT_SWAP(buf[6], buf[8], tmp);
	CMP_GT_SWAP(buf[7], buf[9], tmp);
	CMP_GT_SWAP(buf[10], buf[12], tmp);
	CMP_GT_SWAP(buf[11], buf[13], tmp);
	CMP_GT_SWAP(buf[14], buf[16], tmp);
	CMP_GT_SWAP(buf[15], buf[17], tmp);
	CMP_GT_SWAP(buf[18], buf[20], tmp);
	CMP_GT_SWAP(buf[19], buf[21], tmp);
	CMP_GT_SWAP(buf[22], buf[24], tmp);
	CMP_GT_SWAP(buf[0], buf[1], tmp);
	CMP_GT_SWAP(buf[2], buf[3], tmp);
	CMP_GT_SWAP(buf[4], buf[5], tmp);
	CMP_GT_SWAP(buf[6], buf[7], tmp);
	CMP_GT_SWAP(buf[8], buf[9], tmp);
	CMP_GT_SWAP(buf[10], buf[11], tmp);
	CMP_GT_SWAP(buf[12], buf[13], tmp);
	CMP_GT_SWAP(buf[14], buf[15], tmp);
	CMP_GT_SWAP(buf[16], buf[17], tmp);
	CMP_GT_SWAP(buf[18], buf[19], tmp);
	CMP_GT_SWAP(buf[20], buf[21], tmp);
	CMP_GT_SWAP(buf[22], buf[23], tmp);
	CMP_GT_SWAP(buf[1], buf[16], tmp);
	CMP_GT_SWAP(buf[3], buf[18], tmp);
	CMP_GT_SWAP(buf[5], buf[20], tmp);
	CMP_GT_SWAP(buf[7], buf[22], tmp);
	CMP_GT_SWAP(buf[9], buf[24], tmp);
	CMP_GT_SWAP(buf[1], buf[8], tmp);
	CMP_GT_SWAP(buf[3], buf[10], tmp);
	CMP_GT_SWAP(buf[5], buf[12], tmp);
	CMP_GT_SWAP(buf[7], buf[14], tmp);
	CMP_GT_SWAP(buf[9], buf[16], tmp);
	CMP_GT_SWAP(buf[11], buf[18], tmp);
	CMP_GT_SWAP(buf[13], buf[20], tmp);
	CMP_GT_SWAP(buf[15], buf[22], tmp);
	CMP_GT_SWAP(buf[17], buf[24], tmp);
	CMP_GT_SWAP(buf[1], buf[4], tmp);
	CMP_GT_SWAP(buf[3], buf[6], tmp);
	CMP_GT_SWAP(buf[5], buf[8], tmp);
	CMP_GT_SWAP(buf[7], buf[10], tmp);
	CMP_GT_SWAP(buf[9], buf[12], tmp);
	CMP_GT_SWAP(buf[11], buf[14], tmp);
	CMP_GT_SWAP(buf[13], buf[16], tmp);
	CMP_GT_SWAP(buf[15], buf[18], tmp);
	CMP_GT_SWAP(buf[17], buf[20], tmp);
	CMP_GT_SWAP(buf[19], buf[22], tmp);
	CMP_GT_SWAP(buf[21], buf[24], tmp);
	CMP_GT_SWAP(buf[1], buf[2], tmp);
	CMP_GT_SWAP(buf[3], buf[4], tmp);
	CMP_GT_SWAP(buf[5], buf[6], tmp);
	CMP_GT_SWAP(buf[7], buf[8], tmp);
	CMP_GT_SWAP(buf[9], buf[10], tmp);
	CMP_GT_SWAP(buf[11], buf[12], tmp);
	CMP_GT_SWAP(buf[13], buf[14], tmp);
	CMP_GT_SWAP(buf[15], buf[16], tmp);
	CMP_GT_SWAP(buf[17], buf[18], tmp);
	CMP_GT_SWAP(buf[19], buf[20], tmp);
	CMP_GT_SWAP(buf[21], buf[22], tmp);
	CMP_GT_SWAP(buf[23], buf[24], tmp);
#endif

	filmedCuda[y * width + x] = buf[(k * k + 1) / 2 - 1];

}

int main()
{
	printf("Median filter with k=%i\n", k);
	printf("Image size %i x %i\n", height, width);
	printf("Padded Image size %i x %i\n", heightp, widthp);


	// Variables
	FILE* ini;
	float* G, * filteredImage, * paddedImage;
	char archivo[512] = nameInput;
	int z, x, y;
	cudaError_t cudaerr;

	// Allocate memory
	G = (float*)malloc(height * width * sizeof(float)); // Original image
	filteredImage = (float*)malloc(height * width * sizeof(float));  // Filtered image GPU
	paddedImage = (float*)calloc((heightp) * (widthp), sizeof(float)); // Increase size to (height+k-1) x (width+k-1)

	// Read the image
	if (fopen_s(&ini, archivo, "rb") == 0)
	{
		printf("%s abierto\n", archivo);
	}
	else
	{
		printf("%s fallo al abrirse\n", archivo);
		_getch();
		return(-1);
	}
	fseek(ini, 8L, SEEK_SET); // Offset where the image data starts
	for (z = 0; z < height * width; z++)
	{
		fgetc(ini); // discard red channel
		G[z] = (float)fgetc(ini); // Only save the green channel
		fgetc(ini); // discard blue channel
	}
	fclose(ini);

	// Fill with zeros the extra borders (v+k-1 x h+k-1) + memory coalescing
	for (y = 0; y < height; y++)
	{
		for (x = 0; x < width; x++)
		{
			paddedImage[(y + (k - 1) / 2) * (widthp)+x + (k - 1) / 2] = G[y * width + x];
		}
	}

	// Setup GPU stuff
	cudaEvent_t start, stop;
	float gpu_time;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float* d_src = NULL;
	float* d_dst = NULL;

	cudaMalloc((void**)(&d_src), sizeof(float) * (widthp) * (heightp)); // Padded image
	// Move padded input image from Host to Device
	cudaerr = cudaMemcpy(d_src, paddedImage, sizeof(float) * (heightp) * (widthp), cudaMemcpyHostToDevice);
	if (cudaerr != 0)	printf("ERROR copying paddedImage to d_src (Host to Dev). CudaMalloc value=%i\n\r", cudaerr);
	cudaMalloc((void**)(&d_dst), sizeof(float) * width * height); // Allocate memory for Output image

	// NPP
#if runGPU == 1 || runGPU == 2
	Npp32s xROI, yROI; // First pixel of the ROI in the padded image (top-left corner)
	NppiSize  roi = { width, height }; // Original image size
	NppiSize  mask = { k, k }; // kernel size
	NppiPoint anchor = { (k - 1) / 2, (k - 1) / 2 }; // In the center of the mask
	Npp32u nBufferSize = 0;
	NppStatus status = NPP_SUCCESS;
	Npp32f* pSrcOffset = NULL;

	xROI = (k - 1) / 2; // Set x for initial ROI
	yROI = (k - 1) / 2; // Set y for initial ROI

	Npp32s nSrcStep = sizeof(Npp32f) * (widthp); // width of the padded image. If the parenthesis is removed, all goes to hell, why... no idea!
	Npp32s nDstStep = sizeof(Npp32f) * (width); // Just in case... parenthesis here too. As long the value is multiple of 4 the parenthesis can be removed.
	printf("nScrStep value = %i\n", nSrcStep);
	Npp8u* d_median_filter_buffer = NULL;
	status = nppiFilterMedianGetBufferSize_32f_C1R(roi, mask, &nBufferSize);
	cudaMalloc((void**)(&d_median_filter_buffer), nBufferSize);
	//	pSrcOffset = d_src + yROI*nSrcStep + xROI*PixelSize; // WRONG!!!, Pointers are not handled at byte level.
	pSrcOffset = d_src + yROI * (widthp)+xROI; // 1st pixel of the ROI

	cudaEventRecord(start);
	status = nppiFilterMedian_32f_C1R(pSrcOffset, nSrcStep, d_dst, nDstStep, roi, mask, anchor, d_median_filter_buffer);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpu_time, start, stop);
	printf("\nNPP-GPU Time:  %fms\n\r", gpu_time);
	printf("MedianFilter Speed: %f Megapixels/s\n", (width * height) / gpu_time / 1000);
	printf("Status of nppiFilterMedial_32f_C1R %i (Note: NPP_SUCCESS = 0)\n", status);
	cudaerr = cudaMemcpy(filteredImage, d_dst, sizeof(float) * height * width, cudaMemcpyDeviceToHost);
	if (cudaerr != 0)	printf("ERROR copying d_dst to filteredImage (Dev to Host). CudaMalloc value=%i\n\r", cudaerr);
#endif

	// CUSTOM KERNEL
#if runGPU == 0 || runGPU == 2
	// GPU using naive kernel (based on bs)
	printf("\nGPU Sorting Network version\n");



	cudaFuncSetCacheConfig(mf_gpu_bs, cudaFuncCachePreferL1);

	cudaEventRecord(start);

	// Launch kernels
#if mode == 1
	int threadsPerBlock = 128; // 1D Blocks, 1D Grid
	int numBlocks = width * height / threadsPerBlock; // 1D grid, 1D Block
#endif

#if mode == 2
	dim3 threadsPerBlock(threadsx, threadsy, 1); // Using 2D blocks, 1D grid
	int numBlocks = width * height / (threadsx * threadsy);
#endif

#if mode == 3
	dim3 threadsPerBlock(threadsx, threadsy, 1); // Using 2D blocks, 2D grid
	dim3 numBlocks(width / threadsx, height / threadsy, 1);
#endif

	mf_gpu_bs << <numBlocks, threadsPerBlock >> > (d_dst, d_src);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpu_time, start, stop);
	printf("GPU Time:  %fms\n\r", gpu_time);
	printf("MedianFilter Speed: %f Megapixels/s\n", (width * height) / gpu_time / 1000);
	cudaerr = cudaMemcpy(filteredImage, d_dst, sizeof(float) * height * width, cudaMemcpyDeviceToHost);
	if (cudaerr != 0)	printf("ERROR copying d_dst to filteredImage (Dev to Host). CudaMalloc value=%i\n\r", cudaerr);
#endif

	// Compare vs CPU serial version
	// k4: using a sorting with no conditionals, runs in O(k^2), for a total of O(width*height*k*k)
	// bs: Bubble sort stopping and the middle of the sorting (I only need the median, not the whole sequence sorted)
	// qs: Quicksort generic. Sorts the whole sequence.


	// SERIAL MEDIAN FILTER: We have BS and QS, runCPUSerial=0,1,2,3

#if runCPUSerial != 0 // Use serial run on CPU to compare the GPU results.
	clock_t startCPU;
	clock_t finishCPU;
	float* filteredImageSerial;
	filteredImageSerial = (float*)malloc(height * width * sizeof(float));  // Filtered image CPU serial

#endif

#if runCPUSerial == 1 || runCPUSerial == 3 
	printf("\nCPU using bs:\n");
	startCPU = clock();
	mf_serial_bs(filteredImageSerial, paddedImage); //Median filter with simplified bubble-sort
	finishCPU = clock();
	printf("CPU serial: %fms\n", (double)(finishCPU - startCPU));
	printf("MedianFilter Speed: %f Megapixels/s\n", (width * height) / ((double)(finishCPU - startCPU)) / 1000);
	// verify gpu vs cpu results
	for (z = 0; z < height * width; z++)
	{
		if (filteredImage[z] != filteredImageSerial[z])
		{
			printf("bs: ERROR between CPU and GPU median filters on index: %i\n", z);
			printf("CPU: %f %f,%f %f\n", filteredImageSerial[z], filteredImageSerial[z + 1], filteredImageSerial[z + 2], filteredImageSerial[z + 3]);
			printf("GPU: %f %f,%f %f\n", filteredImage[z], filteredImage[z + 1], filteredImage[z + 2], filteredImage[z + 3]);
			getch();
		}
	}
#endif

#if runCPUSerial == 2 || runCPUSerial == 3 
	printf("\nCPU using qs:\n");
	startCPU = clock();
	mf_serial_qs(filteredImageSerial, paddedImage); //Median filter with quicksort
	finishCPU = clock();
	printf("CPU serial: %fms\n", (double)(finishCPU - startCPU));
	printf("MedianFilter Speed: %f Megapixels/s\n", (width * height) / ((double)(finishCPU - startCPU)) / 1000);
	// verify gpu vs cpu results
	for (z = 0; z < height * width; z++)
	{
		if (filteredImage[z] != filteredImageSerial[z])
		{
			printf("bs: ERROR between CPU and GPU median filters on index: %i\n", z);
			printf("CPU: %f %f,%f %f\n", filteredImageSerial[z], filteredImageSerial[z + 1], filteredImageSerial[z + 2], filteredImageSerial[z + 3]);
			printf("GPU: %f %f,%f %f\n", filteredImage[z], filteredImage[z + 1], filteredImage[z + 2], filteredImage[z + 3]);
			getch();
		}
	}
#endif

	// Move results to an output file (over an existing tiff, just to avoid the mess of creating one)
	FILE* arre = fopen(nameOutput, "rb+");
	if (arre == NULL)
	{
		printf("error en abrir archivo bmw2560Filtered.tiff \n");
	}
	else
	{
		fseek(arre, 8L, SEEK_SET);	// Offset to data
		for (x = 0; x < height; x++)			// Write results in file
		{
			for (y = 0; y < width; y++)
			{
				fputc((unsigned char)filteredImage[x * width + y], arre);
			}
		}
	}
	fclose(arre);

	cudaFree(d_src);
	cudaFree(d_dst);
	free(G);
	free(paddedImage);
	free(filteredImage);

#if runCPUSerial != 0 // Use serial run on CPU to compare the GPU results.
	free(filteredImageSerial);
#endif

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	// Do not close the command window (for Windows), remove for linux
#if pause == 1
	printf("\nAll done... press any key to finish\n");
	getch();
#endif
	return 0;
}




void mf_serial_bs(float* filmed, float* ImBase)
{
	int x, y, a, b, i;
	float buf[k * k], tmp; //main window and temporal window

	for (y = 0; y < height; y++)
	{
		for (x = 0; x < width; x++)
		{
			//Load analysis area (window)
			for (i = 0; i < k * k; i++)
			{
				buf[i] = ImBase[(y + i / k) * (widthp)+x + (i % k)];
			}
			//////////

			for (a = 0; a < (k * k + 1) / 2; a++)
			{
				for (b = a; b < k * k; b++)
				{
					if (buf[a] < buf[b])
					{
						tmp = buf[a];  // swap(buf[a],buf[b]);
						buf[a] = buf[b];
						buf[b] = tmp;
					}
				}
			}
			filmed[y * width + x] = buf[(k * k + 1) / 2 - 1];
		}
	}
}

void mf_serial_qs(float* filmed, float* ImBase)
{
	int x, y, i;
	float buf[k * k]; //main window and temporal window

	for (y = 0; y < height; y++)
	{
		for (x = 0; x < width; x++)
		{
			//Load analysis area (window)
			for (i = 0; i < k * k; i++)
			{
				buf[i] = ImBase[(y + i / k) * (widthp)+x + (i % k)];
			}
			//////////

			quickSort(buf, 0, k * k);
			filmed[y * width + x] = buf[(k * k + 1) / 2 - 1];
		}
	}
}


void quickSort(float* a, int l, int r)
{
	int j;

	if (l < r)
	{
		// divide and conquer
		j = partition(a, l, r);
		quickSort(a, l, j - 1);
		quickSort(a, j + 1, r);
	}

}



int partition(float* a, int l, int r) {
	int i, j;
	float pivot, t;
	pivot = a[l];
	i = l; j = r + 1;

	while (1)
	{
		do ++i; while (a[i] <= pivot && i <= r);
		do --j; while (a[j] > pivot);
		if (i >= j) break;
		t = a[i]; a[i] = a[j]; a[j] = t;
	}
	t = a[l]; a[l] = a[j]; a[j] = t;
	return j;
}
