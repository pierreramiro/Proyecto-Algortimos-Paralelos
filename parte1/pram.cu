#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include<stdlib.h>
#include <time.h>
#ifndef __CUDACC__
#define __CUDACC__
#include <device_functions.h>
#endif

#define N 
#define H 900
#define W 900
#define K 5
#define iterCPU 40

//#define numThreads 
void median_filter_inside(unsigned int* pixel, unsigned int* median_pixel, unsigned int length_col);
unsigned int median_vector(unsigned int* vector);
void median_filter(unsigned int* I_filtered, unsigned int* Image);
int main() {
	unsigned int* Image, * I_filtered;
	//Malloc variables
	Image = (unsigned int*)malloc(H * W * sizeof(unsigned int));
	I_filtered = (unsigned int*)malloc(H * W * sizeof(unsigned int));
	srand(time(NULL));
	for (unsigned int z = 0; z < H * W; z++) {
		Image[z] = rand() % 255;
		//printf("%u\n", Image[z]);
	}
	////Imprimimos la matriz
	//printf("Creamos la matriz\n");
	//for (unsigned int i = 0; i < H; i++) {
	//	for (unsigned int j = 0; j < W; j++) {
	//		printf("%u\t", Image[i * W + j]);
	//	}
	//	printf("\n");
	//}
	//printf("\n");
	////Fin de la impresión de la matriz
	
	/*-----------------------Comienza el código------------------------------*/
	//Iniciamos medición de tiempo
	clock_t startCPU;
	clock_t finishCPU;
	if (K == 0) {
		return 0;
	}
	printf("CPU:\n");
	startCPU = clock();
	for (unsigned int z = 0; z < iterCPU; z++)
	{
		median_filter(I_filtered, Image);
	}
	finishCPU = clock();
	printf("Numero de iteraciones:%d\n", iterCPU);
	printf("Valor de H:%d\tValor de W:%d\tValor de K:%d\t\n", H, W, K);
	printf("CPU serial time: %fms\n", (double)(finishCPU - startCPU)/ iterCPU);/// CLK_TCK);
		
	/*-----------------------Fin el código-------------------------*/

	////Imprimimos la matriz
	//printf("Aplicamos filtro de mediana para K=%d\n", K);
	//for (unsigned int i = 0; i < H; i++) {
	//	for (unsigned int j = 0; j < W; j++) {
	//		printf("%u\t", I_filtered[i * W + j]);
	//	}
	//	printf("\n");
	//}
	////Fin de la impresión de la matriz

	free(Image);
	free(I_filtered);
	return 0;

}

void median_filter_inside(unsigned int* addr_pixel, unsigned int* addr_median_pixel,unsigned int length_col) {
	unsigned int* new_addr = addr_pixel, * vector, index = 0;
	vector = (unsigned int*)malloc(K * K * sizeof(unsigned int));
	//Creamos el nuevo pixel que será el de la esquina superior izquierda de la matriz
	//esto con el fin de realizar un for más cómodo
	new_addr -= (K - 1) / 2 * (1 + length_col);
	for (unsigned int i = 0; i < K; i++) {
		for (unsigned int j = 0; j < K; j++) {
			vector[index] = new_addr[i * length_col + j];
			index++;
		}
	}

	*addr_median_pixel = median_vector(vector);
	free(vector);
}

unsigned int median_vector(unsigned int* vector) {
	//Debemos ordenar los números
	unsigned int count, temp;
	for (unsigned int i = 1; i < K * K; i++) {
		count = i;
		while (vector[count - 1] > vector[count]) {
			temp = vector[count];
			vector[count] = vector[count - 1];
			vector[count - 1] = temp;
			count--;
			if (count == 0) break;
		}
	}
	//procedemos a tomar el número del medio
	return vector[(K * K - 1) / 2];
}


void median_filter(unsigned int* I_filtered,unsigned int* Image) {
	unsigned int z, index;
	unsigned int* Image_pad;
	Image_pad = (unsigned int*)malloc((H + K - 1) * (W + K - 1) * sizeof(unsigned int));

	//Hacemos el padding
		//Primero rellenamos lo interior
	for (unsigned int i = 0; i < H; i++) {
		for (unsigned int j = 0; j < W; j++) {
			Image_pad[(i + (K - 1) / 2) * (W + (K - 1)) + j + (K - 1) / 2] = Image[i * W + j];
		}
	}
	//Llenamos los costados
	for (unsigned int i = 0; i < H; i++) {
		for (unsigned int j = 0; j < (K - 1) / 2; j++) {
			//Izquierda
			Image_pad[((K - 1) / 2 + i) * (W + K - 1) + (K - 1) / 2 - 1 - j] = Image[i * W + j];
			//Derecha
			Image_pad[((K - 1) / 2 + i) * (W + K - 1) + (K - 1) / 2 + W + j] = Image[i * W + W - 1 - j];
		}
	}
	//Llenamos superior e inferior
	for (unsigned int i = 0; i < (K - 1) / 2; i++) {
		for (unsigned int j = 0; j < K - 1 + W; j++) {
			//Fila superior
			Image_pad[((K - 1) / 2 - 1 - i) * (W + K - 1) + j] = Image_pad[((K - 1) / 2 + i) * (W + K - 1) + j];
			//Fila inferior
			Image_pad[((K - 1) / 2 + H + i) * (W + K - 1) + j] = Image_pad[((K - 1) / 2 + H - 1 - i) * (W + K - 1) + j];
		}
	}
	////Imprimimos la matriz con padding
	//for (unsigned int i = 0; i < (H + K - 1); i++) {
	//	for (unsigned int j = 0; j < (W + K - 1); j++) {
	//		printf("%u\t", Image_pad[i * (W + K - 1) + j]);
	//	}
	//	printf("\n");
	//}
	////Fin de la impresión de la matriz

	//Realizamos el çalculo de la mediana
	unsigned int col, row;
	index = 0;
	for (z = 0; z < (H + K - 1) * (W + K - 1); z++) {
		col = z % (W + K - 1);
		row = z / (W + K - 1);
		//Analizamos si está en zona interna
		if ((col > (K - 3) / 2) && (col < W + (K - 1) / 2) && (row > (K - 3) / 2) && (row < H + (K - 1) / 2)) {
			median_filter_inside(&Image_pad[z], &I_filtered[index], (W + K - 1));
			index++;
		}
	}
	free(Image_pad);
}