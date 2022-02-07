#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h> // for rand();
#include <time.h> // to use clock() functions
#ifndef __CUDACC__  
	#define __CUDACC__
	#include <device_functions.h>
#endif

#define ngpu 4194304
#define threadsPerBlock 1024
#define numBlocks 2048
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)

#define iter 10000
#define iterCPU 10000
void eScan(unsigned int* output, unsigned char* input, int length);

__global__ void eScanGPU_up(unsigned int* g_odata, unsigned char* g_idata, int n)
{	//A este punto debemos tener en cuenta que: 
	//1. Debemos crear espacio de memoria para que podamos colocar 2*hilos elementos. 
	//Recordar que cada hilo puede operar con dos elementos.
	__shared__ unsigned int temp[2* threadsPerBlock+64];// allocated on invocation
	//2. Todos los bloques tienen la misma cantidad de hilos
	//3. podemos acceder a elementos especificos de la memoria global con la sgte formula
	// y pasarlos a la shared memory de cada bloque y realizar las operaciones del eScan
	int thid = threadIdx.x + blockIdx.x * blockDim.x;
	int myID = thid + blockIdx.x * blockDim.x;
	int offset = 1;
	//obtemos los indices (el orden) de los elementos a operar en el espacio de memoria
	int ai_g = myID;
	int bi_g = myID + threadsPerBlock;
	int ai = threadIdx.x;
	int bi = threadIdx.x + threadsPerBlock;
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
	temp[ai + bankOffsetA] = g_idata[ai_g];
	temp[bi + bankOffsetB] = g_idata[bi_g];

	for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
	{
		__syncthreads();
		if (threadIdx.x < d)
		{
			int ai = offset * (2 * threadIdx.x + 1) - 1;
			int bi = offset * (2 * threadIdx.x + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	//Nos falta completar la mitad del árbol, pero a este punto este hilo termina su operación con la shared mem
	//Ahora debemos colocar los elementos de la shared en la global para que otro hilo pueda acceder y seguir operando
	//Para ello reseteamos el ai y el bi
	__syncthreads();
	ai = threadIdx.x;
	bi = threadIdx.x + threadsPerBlock;
	g_odata[ai_g] = temp[ai + bankOffsetA];
	g_odata[bi_g] = temp[bi + bankOffsetB];
}

__global__ void eScanGPU_mid(unsigned int* g_odata, unsigned int* g_idata)
{
	__shared__ unsigned int temp[2 * threadsPerBlock + 64];// allocated on invocation
	int thid = threadIdx.x;
	int offset = 1;

	int ai = thid;
	int bi = thid + threadsPerBlock;
	int ai_g = (thid+ 1) * 2 * threadsPerBlock- 1;
	int bi_g = ai_g + ngpu/ 2;
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
	temp[ai + bankOffsetA] = g_idata[ai_g];
	temp[bi + bankOffsetB] = g_idata[bi_g];


	for (int d = numBlocks >> 1; d > 0; d >>= 1) // build sum in place up the tree
	{
		__syncthreads();
		if (thid < d)
		{
			int ai = offset * (2 * thid + 1) - 1;
			int bi = offset * (2 * thid + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);
			//			int ai = offset*(2*thid+1)-1;
			//			int bi = offset*(2*thid+2)-1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	if (thid == 0) { temp[numBlocks - 1 + CONFLICT_FREE_OFFSET(numBlocks - 1)] = 0; }
	//	if (thid == 0) { temp[n - 1] = 0; } // clear the last element
	for (int d = 1; d < numBlocks; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (thid < d)
		{
			int ai = offset * (2 * thid + 1) - 1;
			int bi = offset * (2 * thid + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);
			//			int ai = offset*(2*thid+1)-1;
			//			int bi = offset*(2*thid+2)-1;

			unsigned int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();
	g_odata[ai_g] = temp[ai + bankOffsetA];
	g_odata[bi_g] = temp[bi + bankOffsetB];
}

__global__ void eScanGPU_down(unsigned int* g_odata, unsigned int* g_idata, int n)
{
	__shared__ unsigned int temp[2 * threadsPerBlock + 64];// allocated on invocation
	int thid = threadIdx.x + blockIdx.x * blockDim.x;
	int myID = thid + blockIdx.x * blockDim.x;
	int offset = 1;
	int ai_g = myID;
	int bi_g = myID + threadsPerBlock;
	int ai = threadIdx.x;
	int bi = threadIdx.x + (n / 2);
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
	temp[ai + bankOffsetA] = g_idata[ai_g];
	temp[bi + bankOffsetB] = g_idata[bi_g];

	//	temp[2*thid] = g_idata[2*thid]; // load input into shared memory
	//	temp[2*thid+1] = g_idata[2*thid+1];
	for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
	{
		offset *= 2;
	}
	for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (threadIdx.x < d)
		{
			int ai = offset * (2 * threadIdx.x + 1) - 1;
			int bi = offset * (2 * threadIdx.x + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);
			//			int ai = offset*(2*thid+1)-1;
			//			int bi = offset*(2*thid+2)-1;

			unsigned int t= temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();
	g_odata[ai_g] = temp[ai + bankOffsetA];
	g_odata[bi_g] = temp[bi + bankOffsetB];
}



int main()
{
    unsigned char* in;
    unsigned int* outgpu, * outcpu;
    cudaError_t cudaerr;
    int z;

    in = (unsigned char*)malloc(ngpu * sizeof(unsigned char)); // input data
    outgpu = (unsigned int*)malloc(ngpu * sizeof(unsigned int)); // output data
    outcpu = (unsigned int*)malloc(ngpu * sizeof(unsigned int)); // output data
    // Fill data
	srand(time(NULL));
    for (z = 0; z < ngpu; z++)
    {
        in[z] = rand() % 256; // Numbers between 0 and 255
    }

    unsigned char* d_src_up = NULL;
    unsigned int* d_dst_up = NULL;
	unsigned int* d_dst_mid = NULL;
	unsigned int* d_dst_down = NULL;

    cudaMalloc((void**)(&d_src_up), sizeof(unsigned char) * ngpu); // Input data
    // Move padded input image from Host to Device
    cudaerr = cudaMemcpy(d_src_up, in, sizeof(unsigned char) * ngpu, cudaMemcpyHostToDevice);
    if (cudaerr != 0)	printf("ERROR copying in data to d_src (Host to Dev). CudaMalloc value=%i\n\r", cudaerr);
    cudaMalloc((void**)(&d_dst_up), sizeof(unsigned int) * ngpu); // Output data
	cudaMalloc((void**)(&d_dst_mid), sizeof(unsigned int) * ngpu); // Output data
	cudaMalloc((void**)(&d_dst_down), sizeof(unsigned int) * ngpu); // Output data

	
	cudaFuncSetCacheConfig(eScanGPU_up, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(eScanGPU_mid, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(eScanGPU_down, cudaFuncCachePreferL1);

	// Setup timing using cudaEvent
	cudaEvent_t start, stop;
	float gpu_time;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	printf("\nGPU eScan\n");

	cudaEventRecord(start);
	//WARM UP
	//ejecutamos el Kernel
	eScanGPU_up << <numBlocks, threadsPerBlock >> > (d_dst_up, d_src_up, numBlocks);
	cudaDeviceSynchronize();
	cudaerr = cudaMemcpy(d_dst_mid, d_dst_up, sizeof(unsigned int) * ngpu, cudaMemcpyDeviceToDevice);
	if (cudaerr != 0)	printf("ERROR copying in data to d_src (Host to Dev). CudaMalloc value=%i\n\r", cudaerr);
	eScanGPU_mid << <1, threadsPerBlock >> > (d_dst_mid, d_dst_up);
	cudaDeviceSynchronize();
	eScanGPU_down << <numBlocks, threadsPerBlock >> > (d_dst_down, d_dst_mid, numBlocks);
	cudaDeviceSynchronize();
	cudaerr = cudaMemcpy(outgpu, d_dst_down, sizeof(unsigned int) * ngpu, cudaMemcpyDeviceToHost);
	if (cudaerr != 0)	printf("ERROR copying d_dst to outgpu (Dev to Host). CudaMalloc value=%i\n\r", cudaerr);
	//Fin del kernel
	for (z = 0; z < iter; z++)
	{
		eScanGPU_up << <numBlocks, threadsPerBlock >> > (d_dst_up, d_src_up, numBlocks);
		cudaDeviceSynchronize();
		cudaerr = cudaMemcpy(d_dst_mid, d_dst_up, sizeof(unsigned int) * ngpu, cudaMemcpyHostToDevice);
		if (cudaerr != 0)	printf("ERROR copying in data to d_src (Host to Dev). CudaMalloc value=%i\n\r", cudaerr);
		eScanGPU_mid << <1, threadsPerBlock >> > (d_dst_mid, d_dst_up);
		cudaDeviceSynchronize();
		eScanGPU_down << <numBlocks, threadsPerBlock >> > (d_dst_down, d_dst_mid, numBlocks);
		cudaDeviceSynchronize();
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpu_time, start, stop);
	printf("GPU Time:  %fms\n\r", gpu_time / iter);
	printf("eScanGPU Speed: %f MegaOps/s\n", ngpu / (gpu_time / iter) / 1000);
	cudaerr = cudaMemcpy(outgpu, d_dst_down, sizeof(unsigned int) * ngpu, cudaMemcpyDeviceToHost);
	if (cudaerr != 0)	printf("ERROR copying d_dst to outgpu (Dev to Host). CudaMalloc value=%i\n\r", cudaerr);


	clock_t startCPU;
	clock_t finishCPU;

	printf("\nCPU using eScan:\n");
	startCPU = clock();
	for (z = 0; z < iterCPU; z++)
	{
		eScan(outcpu, in, ngpu);
	}
	finishCPU = clock();
	printf("CPU serial: %fms\n", (double)(finishCPU - startCPU) / 1000 / iterCPU);/// CLK_TCK);
	printf("eScanCPU Speed: %f MegaOps/s\n", ngpu / ((double)(finishCPU - startCPU)) / 1000 * iterCPU);
	// verify gpu vs cpu results
	for (z = 0; z < ngpu; z++)
	{
		if (outgpu[z] != outcpu[z])
		{
			//error += abs(filteredImage[z] - filteredImageSerial[z]);
			printf("ERROR between CPU and GPU Scan on index: %i\n", z);
			printf("CPU: %u %u,%u %u\n", outcpu[z], outcpu[z + 1], outcpu[z + 2], outcpu[z + 3]);
			printf("GPU: %u %u,%u %u\n", outgpu[z], outgpu[z + 1], outgpu[z + 2], outgpu[z + 3]);
		}
	}

	printf("\n All DONE, press any key to end");
    cudaFree(d_src_up);
    cudaFree(d_dst_up); 
	cudaFree(d_dst_mid); 
	cudaFree(d_dst_down);
    free(in);
    free(outcpu);
    free(outgpu);
    return 0;
}

void eScan(unsigned int* output, unsigned char* input, int length)
{

    output[0] = 0;
    for (int z = 1; z < length; ++z)
    {
        output[z] = input[z - 1] + output[z - 1];
    }
}

