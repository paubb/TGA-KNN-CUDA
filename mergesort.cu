#include <stdlib.h>
#include <stdio.h>

//AQUEST ÉS EL CODI PROPORCIONAT PELS NOSTRES COMPANYS

__device__ void mergeDevice(int *list, int *sorted, int start, int mid, int end)
{
    int ti=start, i=start, j=mid;
    while (i<mid || j<end)
    {
        if (j==end) sorted[ti] = list[i++];
        else if (i==mid) sorted[ti] = list[j++];
        else if (list[i]<list[j]) sorted[ti] = list[i++];
        else sorted[ti] = list[j++];
        ti++;
    }

    for (ti=start; ti<end; ti++)
        list[ti] = sorted[ti];
}

void mergeHost(int *list, int *sorted, int start, int mid, int end)
{
    int ti=start, i=start, j=mid;
    while (i<mid || j<end)
    {
        if (j==end) sorted[ti] = list[i++];
        else if (i==mid) sorted[ti] = list[j++];
        else if (list[i]<list[j]) sorted[ti] = list[i++];
        else sorted[ti] = list[j++];
        ti++;
    }

    for (ti=start; ti<end; ti++)
        list[ti] = sorted[ti];
}

__device__ void mergeSortKernel(int *list, int *sorted, int start, int end)
{   
    //Final 1: hi ha mes threads que elements del vector
    if (end-start<2)
        return;
  
    mergeSortKernel(list, sorted, start, start + (end-start)/2);
    mergeSortKernel(list, sorted, start + (end-start)/2, end);
    mergeDevice(list, sorted, start, start + (end-start)/2, end);
}

__global__ void callMerge(int *list, int *sorted, int chunkSize, int N) {
  	if (chunkSize >= N)
		return;	
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	int start = tid*chunkSize;
	int end = start + chunkSize;
	if (end > N) {
		end = N;
	}
	mergeDevice(list, sorted, start, start + (end-start)/2, end);
}

__global__ void callMergeSort(int *list, int *sorted, int chunkSize, int N) {
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	int start = tid*chunkSize;
	int end = start + chunkSize;
	if (end > N) {
		end = N;
	}
	mergeSortKernel(list, sorted, start, end);
}

void printArray(int A[], int size) 
{ 
    int i; 
    for (i=0; i < size; i++) 
        printf("%d ", A[i]); 
    printf("\n"); 
}

int contarSeparacions(int A[], int size) 
{ 	
    int s = 0;
    int i; 
    for (i=0; i < size-1; i++) {
	if (A[i] > A[i+1]) {
	  s++;
	}
    }
    return s;
}

void sortBlocks(int *list, int *sorted, int N, int s) {
  int chunkSize = N/s;
  int start = 0;
  int end = chunkSize;
  int mid = (start+end)/2;
	
  mergeHost(list, sorted, start, mid, end);
  
/*
  while (chunkSize < (N + chunkSize)) {
	mergeHost(list, sorted, start, mid, end);
	chunkSize = chunkSize*2;
	end = end + chunkSize;
	mid = end - chunkSize;
  }
*/
}


void InitV(int N, int *v) {
   int i;
   for (i=0; i<N; i++) 
     v[i] = rand() % 131072;  
}

int main() {
    int *arr_h, *arrSorted_h, *arrSortedF_h;
    int *arr_d, *arrSorted_d, *arrSortedF_d;
    int chunkSize;
    unsigned int nBytes;
    unsigned int N; 
    unsigned int nBlocks, nThreads;
   
    N = 131072;
    nThreads = 128;
    nBlocks = 32;
    chunkSize = N/(nThreads*nBlocks);

    nBytes = N * sizeof(int);

    cudaEvent_t start, stop;
    float timeTaken;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    arr_h = (int*) malloc(nBytes);
    arrSorted_h = (int*) malloc(nBytes);
    arrSortedF_h = (int*) malloc(nBytes);

    cudaMallocHost((int **) &arr_h, nBytes);
    cudaMallocHost((int **) &arrSorted_h, nBytes);
    cudaMallocHost((int **) &arrSortedF_h, nBytes);

    InitV(N, arr_h);

    cudaMalloc((int**)&arr_d, nBytes);
    cudaMalloc((int**)&arrSorted_d, nBytes);
    cudaMalloc((int**)&arrSortedF_d, nBytes);

    cudaMemcpy(arr_d, arr_h, nBytes, cudaMemcpyHostToDevice);
    
    printf("Given array is \n"); 
    printArray(arr_h, N);
    
    cudaEventRecord(start, 0);
    callMergeSort<<<nBlocks, nThreads>>>(arr_d, arrSorted_d,chunkSize, N);
    int auxChunkSize = chunkSize*2;
    int auxBlock = nBlocks;
    int auxThread = nThreads/2;
    while (auxChunkSize < N) {
       callMerge<<<auxBlock, auxThread>>>(arrSorted_d, arrSortedF_d, auxChunkSize, N);
       auxChunkSize = auxChunkSize*2;
       //auxThread = auxThread/2;
    }
    cudaMemcpy(arrSorted_h, arrSortedF_d, nBytes, cudaMemcpyDeviceToHost);
    
    int s = contarSeparacions(arrSorted_h, N);
    printf("\nSEPARACIONS: %d \n", s);
    sortBlocks(arrSorted_h, arrSortedF_h, N, s);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaFree(arr_d);
    cudaFree(arrSorted_d);
    cudaFree(arrSortedF_d);

    cudaEventElapsedTime(&timeTaken,  start, stop);
    
    printf("\nSorted array is \n"); 
    printArray(arrSortedF_h, N); 
    
    printf("SEPARACIONS: %d\n", s);
    printf("nThreads: %d\n", nThreads);
    printf("nBlocks: %d\n", nBlocks);
    printf("Tiempo Total %4.6f ms\n", timeTaken);
    printf("Ancho de Banda %4.3f GB/s\n", (N * sizeof(int)) / (1000000 * timeTaken));
    return 0; 
    
}
