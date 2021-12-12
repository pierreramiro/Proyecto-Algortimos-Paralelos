#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include<stdlib.h>
#include <time.h>
#ifndef __CUDACC__
#define __CUDACC__
#include <device_functions.h>
#endif


#define H 23
#define W 13 
#define K 5

//#define numThreads 
void median_filter_inside(unsigned int* pixel, unsigned int* median_pixel, unsigned int length_col);
unsigned int median_vector(unsigned int* vector);
int main() {
	unsigned int* Image, * I_filtered, * Image_pad;
	unsigned int z, index;
	//Malloc variables
	Image = (unsigned int*)malloc(H * W * sizeof(unsigned int));
	I_filtered = (unsigned int*)malloc(H * W * sizeof(unsigned int));
	Image_pad= (unsigned int*)malloc((H + K - 1) * (W + K - 1) * sizeof(unsigned int));
	srand(time(NULL));
	for (z = 0; z < H * W; z++) {
		Image[z] = rand() % 255;
		//printf("%u\n", Image[z]);
	}
	//Imprimimos la matriz
	for (unsigned int i = 0; i < H; i++) {
		for (unsigned int j = 0; j < W; j++) {
			printf("%u\t", Image[i * W + j]);
		}
		printf("\n");
	}
	printf("\n");
	//Fin de la impresi�n de la matriz
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
			Image_pad[((K - 1) / 2 + i ) * (W + K - 1) + (K - 1) / 2 -1- j] = Image[i* W + j];
			//Derecha
			Image_pad[((K-1)/2+ i) * (W + K - 1) +  (K - 1) /2 + W +j] = Image[i*W + W - 1- j];
		}
	}
	//Llenamos superior e inferior
	for (unsigned int i = 0; i < (K - 1) / 2; i++) {
		for (unsigned int j = 0; j < K-1+W; j++) {
			//Fila superior
			Image_pad[((K-1)/2-1-i)*(W + K - 1) + j] = Image_pad[((K-1)/2+i)*(W+K-1) + j];
			//Fila inferior
			Image_pad[((K-1)/2+H+i)* (W + K - 1) + j] = Image_pad[((K-1)/2+H-1-i) * (W + K - 1) +j];
		}
	}	
	////Imprimimos la matriz con padding
	//for (unsigned int i = 0; i < (H + K - 1); i++) {
	//	for (unsigned int j = 0; j < (W + K - 1); j++) {
	//		printf("%u\t", Image_pad[i * (W + K - 1) + j]);
	//	}
	//	printf("\n");
	//}
	////Fin de la impresi�n de la matriz

	//Realizamos el �alculo de la mediana
	if (K == 0) {
		return 0;
	}
	unsigned int col, row;
	index = 0;
	for (z = 0; z < (H+K-1)*(W+K-1); z++) {
		col = z % (W + K - 1);
		row = z / (W + K - 1);
		//Analizamos si est� en zona interna
		if ((col > (K - 3) / 2) && (col < W + (K-1)/2) && (row > (K - 3) / 2) && (row < H + (K - 1) / 2)) {
			median_filter_inside(&Image_pad[z], &I_filtered[index], (W + K - 1));
			index ++;
		}
	}
	//Imprimimos la matriz
	for (unsigned int i = 0; i < H; i++) {
		for (unsigned int j = 0; j < W; j++) {
			printf("%u\t", I_filtered[i * W + j]);
		}
		printf("\n");
	}
	//Fin de la impresi�n de la matriz
	return 0;
}

void median_filter_inside(unsigned int* addr_pixel, unsigned int* addr_median_pixel,unsigned int length_col) {
	unsigned int* new_addr = addr_pixel, * vector, index = 0;
	vector = (unsigned int*)malloc(K * K * sizeof(unsigned int));
	//Creamos el nuevo pixel que ser� el de la esquina superior izquierda de la matriz
	//esto con el fin de realizar un for m�s c�modo
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
	//Debemos ordenar los n�meros
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
	//procedemos a tomar el n�mero del medio
	return vector[(K * K - 1) / 2];
}