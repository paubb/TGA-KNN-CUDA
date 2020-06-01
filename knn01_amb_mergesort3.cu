#include <stdio.h>
#include <stdlib.h>

#include <math.h>
#include <iostream>     // std::cout
#include <algorithm>    // std::sort
#include <vector>       // std::vector
#include <time.h>
using namespace std;

#define PINNED 0
#define THREADS 1024

__device__ void mergeDevice(float *list, float *sorted, float *list2, float *sorted2, int start, int mid, int end)
{
    int ti=start, i=start, j=mid;
    while (i<mid || j<end)
    {
        if (j==end) {
            sorted[ti] = list[i++];
            sorted2[ti] = list2[i++];
        }
        else if (i==mid) {
            sorted[ti] = list[j++];
            sorted2[ti] = list2[j++];
        }
        else if (list[i]<list[j]) {
         sorted[ti] = list[i++];
        sorted2[ti] = list2[i++];
    }
        else {
        sorted[ti] = list[j++];
        sorted2[ti] = list2[j++];
    }
        ti++;
    }

    for (ti=start; ti<end; ti++) {
        list[ti] = sorted[ti];
        list2[ti] = sorted2[ti];
    }
}

__device__ void mergeSortKernel(float *list, float *sorted, float *list2, float *sorted2, int start, int end)
{
    //Final 1: hi ha mes threads que elements del vector
    if (end-start<2)
        return;

    mergeSortKernel(list, sorted, list2, sorted2, start, start + (end-start)/2);
    mergeSortKernel(list, sorted, list2, sorted2, start + (end-start)/2, end);
    mergeDevice(list, sorted, list2, sorted2, start, start + (end-start)/2, end);
}

__global__ void callMerge(float *list, float *sorted, float *list2, float *sorted2, int chunkSize, int N) {
      if (chunkSize >= N)
        return;
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int start = tid*chunkSize;
    int end = start + chunkSize;
    if (end > N) {
        end = N;
    }
    mergeDevice(list, sorted, list2, sorted2, start, start + (end-start)/2, end);
}

__global__ void callMergeSort(float *list, float *sorted, float *list2, float *sorted2, int chunkSize, int N) {
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int start = tid*chunkSize;
    int end = start + chunkSize;
    if (end > N) {
        end = N;
    }
    mergeSortKernel(list, sorted, list2, sorted2, start, end);
}

struct Point
{
    float x, y;     // Co-ordinate of point
};


// A utility function to swap two elements
void swap(float* a, float* b, float* c, float* d)
{
    int t = *a;
    int t2 = *c;
    *a = *b;
    *c = *d;
    *b = t;
    *d = t2;
}

/* This function takes last element as pivot, places
   the pivot element at its correct position in sorted
    array, and places all smaller (smaller than pivot)
   to left of pivot and all greater elements to right
   of pivot */
int partition (float *result_prediction_host, float *ref_points_host_val, int low, int high)
{
    int pivot = result_prediction_host[high];    // pivot
    int i = (low - 1);  // Index of smaller element

    for (int j = low; j <= high- 1; j++)
    {
        // If current element is smaller than or
        // equal to pivot
        if (result_prediction_host[j] <= pivot)
        {
            i++;    // increment index of smaller element
            swap(&result_prediction_host[i], &result_prediction_host[j], &ref_points_host_val[i], &ref_points_host_val[j]);
        }
    }
    swap(&result_prediction_host[i + 1], &result_prediction_host[high], &ref_points_host_val[i + 1], &ref_points_host_val[high]);
    return (i + 1);
}

/* The main function that implements QuickSort
 arr[] --> Array to be sorted,
  low  --> Starting index,
  high  --> Ending index */
void quickSort(float *result_prediction_host, float *ref_points_host_val, int low, int high)
{
    if (low < high)
    {
        /* pi is partitioning index, arr[p] is now
           at right place */
        int pi = partition(result_prediction_host, ref_points_host_val, low, high);

        // Separately sort elements before
        // partition and after partition
        quickSort(result_prediction_host, ref_points_host_val, low, pi - 1);
        quickSort(result_prediction_host, ref_points_host_val, pi + 1, high);
    }
}

/**
 * @param arr    refence points
 * @param n      number of reference points
 * @param k      number of points we want to use for the prediction
 * @param p      point we want to predict
 */
int classifyAPoint(Point arr[], int n, int k, Point p, float val[])
{
    float distances[n];

    // Fill distances of all points from p
    for (int i = 0; i < n; i++)
        distances[i] =
            sqrt((arr[i].x - p.x) * (arr[i].x - p.x) +
                 (arr[i].y - p.y) * (arr[i].y - p.y));

    // Sort the Points by distance from p
    quickSort(distances, val, 0, n-1);

    // Now consider the first k elements and only
    // two groups
    int freq1 = 0;     // Frequency of group 0
    int freq2 = 0;     // Frequency of group 1
    for (int i = 0; i < k; i++)
    {
        if (val[i] == 0)
            freq1++;
        else if (val[i] == 1)
            freq2++;
    }
    printf ("freq1 is %d.\n", freq1);
    printf ("freq2 is %d.\n", freq2);

    return (freq1 > freq2 ? 0 : 1);
}

void InitHostInput(Point arr[], int n, float val[], Point p, float *ref_points_host_x, float *ref_points_host_y, float *ref_points_host_val) {

    for (int i=0; i<n; i++) {
        ref_points_host_x[i] = arr[i].x;
        ref_points_host_y[i] = arr[i].y;
        ref_points_host_val[i] = val[i];
    }

}

void InitHostFreq(unsigned int *freq_host) {

    freq_host[0] = 0;
    freq_host[1] = 0;

}

__global__ void calculateDistance(Point p, float *ref_points_dev_x, float *ref_points_dev_y, float *result_prediction_dev) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Fill distances of all points from p

        result_prediction_dev[i] =
            sqrt((ref_points_dev_x[i] - p.x) * (ref_points_dev_x[i] - p.x) +
                 (ref_points_dev_y[i] - p.y) * (ref_points_dev_y[i] - p.y));


}

__global__ void calculateFreq(float *ref_points_host_val, unsigned int *freq_dev) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;


        int j = ref_points_host_val[i];
            atomicAdd(&freq_dev[j], 1);
}

int classifyAPointCUDA(Point arr[], float val[], int n, int k, Point p)
{

    unsigned int numBytes;
    unsigned int nBlocks, nThreads;

    int chunkSize_sort;
    unsigned int nBytes_sort;
    unsigned int nBlocks_sort, nThreads_sort;

    float TiempoKernelDistance, TiempoSort, TiempoKernelFreq, TiempoAllOperations;
    cudaEvent_t E0, E1, E2, E3, E4, E5;

    float *ref_points_dev_x   = NULL;
    float *ref_points_dev_y   = NULL;
    float *ref_points_dev_val   = NULL;
    float *result_prediction_dev  = NULL;

    float *ref_points_host_x   = NULL;
    float *ref_points_host_y = NULL;
    float *ref_points_host_val   = NULL;
    float *result_prediction_host  = NULL;

    float *arrSorted_h, *arrSortedF_h;
    float *arrSorted_d, *arrSortedF_d;
    float *arrSorted2_h, *arrSortedF2_h;
    float *arrSorted2_d, *arrSortedF2_d;

    unsigned int *freq_dev = NULL;
    unsigned int *freq_host = NULL;


    // numero de Threads
    nThreads = THREADS;

    // numero de Blocks en cada dimension
    nBlocks = (n+nThreads-1)/nThreads;
    printf("nBlocks = %d \n", nBlocks);

    numBytes = nBlocks * nThreads * sizeof(float);
    printf("numBytes = %d \n", numBytes);

    nThreads_sort = 128;
    nBlocks_sort = 32;
    chunkSize_sort = n/(nThreads_sort*nBlocks_sort);
    nBytes_sort = n * sizeof(float);

    cudaEventCreate(&E0);
    cudaEventCreate(&E1);
    cudaEventCreate(&E2);
    cudaEventCreate(&E3);
    cudaEventCreate(&E4);
    cudaEventCreate(&E5);

    if (PINNED) {
        // Obtiene Memoria [pinned] en el host
        cudaMallocHost((float**)&ref_points_host_x, numBytes);
        cudaMallocHost((float**)&ref_points_host_y, numBytes);
        cudaMallocHost((float**)&ref_points_host_val, nBytes_sort);
        cudaMallocHost((float**)&result_prediction_host, nBytes_sort);

        cudaMallocHost((float**)&arrSorted_h, nBytes_sort);
        cudaMallocHost((float**)&arrSortedF_h, nBytes_sort);

        cudaMallocHost((float**)&freq_host, sizeof(unsigned int)*2);


    } else {
        // Obtener Memoria en el host
        ref_points_host_x = (float*) malloc(numBytes);
        ref_points_host_y = (float*) malloc(numBytes);
        ref_points_host_val = (float*) malloc(nBytes_sort);
        result_prediction_host = (float*) malloc(nBytes_sort);

        arrSorted_h = (float*) malloc(nBytes_sort);
        arrSortedF_h = (float*) malloc(nBytes_sort);
        arrSorted2_h = (float*) malloc(nBytes_sort);
        arrSortedF2_h = (float*) malloc(nBytes_sort);

        freq_host = (unsigned int*) malloc(sizeof(unsigned int)*2);
    }

    InitHostInput(arr, n, val, p, ref_points_host_x, ref_points_host_y, ref_points_host_val);

    InitHostFreq(freq_host);

    // Obtener Memoria en el device
    cudaMalloc((float**)&ref_points_dev_x, numBytes);
    cudaMalloc((float**)&ref_points_dev_y, numBytes);
    cudaMalloc((float**)&ref_points_dev_val, nBytes_sort);
    cudaMalloc((float**)&result_prediction_dev, nBytes_sort);

    cudaMalloc((float **) &arrSorted_d, nBytes_sort);
    cudaMalloc((float **) &arrSortedF_d, nBytes_sort);
    cudaMalloc((float **) &arrSorted2_d, nBytes_sort);
    cudaMalloc((float **) &arrSortedF2_d, nBytes_sort);

    cudaMalloc((unsigned int**)&freq_dev, sizeof(unsigned int)*2);

    // Copiar datos desde el host en el device
    cudaMemcpy(ref_points_dev_x, ref_points_host_x, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(ref_points_dev_y, ref_points_host_y, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(ref_points_dev_val, ref_points_host_val, nBytes_sort, cudaMemcpyHostToDevice);

    cudaMemcpy(freq_dev, freq_host, sizeof(unsigned int)*2, cudaMemcpyHostToDevice);

    cudaEventRecord(E0, 0);

    // Ejecutar el kernel
    calculateDistance<<<nBlocks, nThreads>>>(p, ref_points_dev_x, ref_points_dev_y, result_prediction_dev);

    cudaEventRecord(E1, 0); cudaEventSynchronize(E1);
    cudaEventElapsedTime(&TiempoKernelDistance,  E0, E1);

    // Obtener el resultado desde el host
    //cudaMemcpy(result_prediction_host, result_prediction_dev, numBytes, cudaMemcpyDeviceToHost);

    // Liberar Memoria del device
    cudaFree(ref_points_dev_x);
    cudaFree(ref_points_dev_y);


    cudaEventRecord(E4, 0);
    // Sort the Points by distance from p

    printf("Invocació Kernel Sort <<<nBlocks, nKernels>>> (N): <<<%d, %d>>> (%d)\n", nBlocks_sort, nThreads_sort, n);

    callMergeSort<<<nBlocks_sort, nThreads_sort>>>(result_prediction_dev, arrSorted_d, ref_points_dev_val, arrSorted2_d, chunkSize_sort, n);
    int auxChunkSize = chunkSize_sort*2;
    int auxBlock = nBlocks_sort;
    int auxThread = nThreads_sort/2;

    cudaFree(result_prediction_dev);
    cudaFree(ref_points_dev_val);

    while (auxChunkSize < n) {
        //printf("Invocació Kernel Sort 2 <<<nBlocks, nKernels>>> (N): <<<%d, %d>>> (%d)\n", auxBlock, auxThread, n);
       callMerge<<<auxBlock, auxThread>>>(arrSorted_d, arrSortedF_d, arrSorted2_d, arrSortedF2_d, auxChunkSize, n);
       auxChunkSize = auxChunkSize*2;
       auxThread = auxThread/2;
    }

    //cudaMemcpy(arrSorted_h, arrSortedF_d, nBytes_sort, cudaMemcpyDeviceToHost);

    cudaFree(arrSorted_d);
    cudaFree(arrSortedF_d);

    //quickSort(result_prediction_host, ref_points_host_val, 0, n-1);
    /*
        for(int i = 0; i < n; i++){
        printf("L'element: %d\n", i);
        printf("La distancia: %f\n", result_prediction_host[i]);
        printf("La x: %f\n", ref_points_host_val[i]);
    }
    */

    cudaEventRecord(E5, 0); cudaEventSynchronize(E5);
    cudaEventElapsedTime(&TiempoSort,  E4, E5);

    cudaEventRecord(E2, 0);

    // Ejecutar el kernel
    calculateFreq<<<k, 1>>>(arrSortedF2_d, freq_dev);

    cudaEventRecord(E3, 0); cudaEventSynchronize(E3);
    cudaEventElapsedTime(&TiempoKernelFreq,  E2, E3);

    TiempoAllOperations = TiempoKernelDistance + TiempoSort + TiempoKernelFreq;

    cudaMemcpy(freq_host, freq_dev, sizeof(unsigned int)*2, cudaMemcpyDeviceToHost);


    cudaFree(ref_points_dev_val);
    cudaFree(freq_dev);


    int result = -1;
    if(freq_host[0] > freq_host[1]) result = 0;
    else result = 1;

    printf ("freq1 is %d.\n", freq_host[0]);
    printf ("freq2 is %d.\n", freq_host[1]);

    printf ("The value classified to unknown point"
            " is %d.\n", result);


    printf("Invocació Kernel <<<nBlocks, nKernels>>> (N): <<<%d, %d>>> (%d)\n", nBlocks, nThreads, n);

    printf("Tiempo Kernel calculo distancia (00): %4.6f milseg\n", TiempoKernelDistance);
    printf("Tiempo Kernel calculo freq (00): %4.6f milseg\n", TiempoKernelFreq);
    printf("Tiempo Sort (00): %4.6f milseg\n", TiempoSort);
    printf("Tiempo todas las operaciones (00): %4.6f milseg\n", TiempoAllOperations);

    if (PINNED) printf("Usando Pinned Memory\n");
    else printf("NO usa Pinned Memory\n");

    if (PINNED) {
        cudaFreeHost(ref_points_host_x); cudaFreeHost(ref_points_host_y); cudaFreeHost(ref_points_host_val);
        cudaFreeHost(result_prediction_host); cudaFreeHost(freq_host); cudaFreeHost(arrSorted_h);cudaFreeHost(arrSortedF_h); cudaFreeHost(arrSorted_h);cudaFreeHost(arrSortedF_h);
    } else {
        free(ref_points_host_x); free(ref_points_host_y); free(ref_points_host_val); free(result_prediction_host);
        free(arrSorted_h); free(arrSortedF_h); free(freq_host); free(arrSorted2_h); free(arrSortedF2_h);
    }


    return result;

}

void InitKDefecte(int *k) {

    // Parameter to decide group of the testing point
    (*k) = 15;

}

void InitTestPointDefecte(struct Point *p) {

    //Test Point
    p->x = 2.5;
    p->y = 7;

}

void InitDefecte(int *k, struct Point *p) {
    InitKDefecte(k);
    InitTestPointDefecte(p);
}

int main(int argc, char** argv)
{
    srand(time(0));

    //Es declaren les variables
    int n, k;
    struct Point p;

    //S'inicialitza la K, i les coordenades del Testing point
    if (argc == 1)      { InitDefecte(&k, &p); }
    else if (argc == 2) { k = atoi(argv[1]); InitTestPointDefecte(&p); }
    else if (argc == 4) { k = atoi(argv[1]); p.x = atof(argv[2]); p.y = atof(argv[3]);}
    else { printf("Usage: ./exe k TestPointCoordenadaX TestPointCoordenadaY\n"); exit(0); }

    //Es crea l'estructura sobre la qual es vol fer la predicció
    n = 131072; // Number of data points
    Point arr[n];

    float val_seq[n];
    float val_cuda[n];

    for(int i = 0; i < n; ++i) {
        arr[i].x = rand();
        arr[i].y = rand();
        val_seq[i] = rand() % 2;
        val_cuda[i] = val_seq[i];
    }

    /*for(int i = 0; i < n; i++){
        printf("x: %lf\n", arr[i].x);
        printf("y: %lf\n", arr[i].y);
        printf("val: %f\n", val[i]);
    }*/

    printf("k = %d \n", k);

    printf("The Testing Point values are:");
    printf(" x = %f", p.x);
    printf(" and");
    printf(" y = %f", p.y);
    printf("\n");

    printf("\n");
    printf("Programa Seqüencial -------------------------------------------------- \n");
    printf("\n");

    // Calculate the time taken by the sequential code: classifyAPoint function
    clock_t t;
    t = clock();
    int result = classifyAPoint(arr, n, k, p, val_seq);
    t = clock() - t;
    float time_taken = ((float)t)/(CLOCKS_PER_SEC/1000); // in mseconds

    printf ("The value classified to unknown point"
            " is %d.\n", result);

    printf ("Temps total seqüencial:"
            " %lf milseg.\n", time_taken);

    printf("\n");
    printf("Programa CUDA -------------------------------------------------------- \n");
    printf("\n");

    // Calculate the time taken by the sequential code: classifyAPoint function
    clock_t t2;
    t2 = clock();
    int result2 = classifyAPointCUDA(arr,val_cuda, n, k, p);
    t2 = clock() - t2;
    float time_taken2 = ((float)t2)/(CLOCKS_PER_SEC/1000); // in mseconds

    printf ("The value classified to unknown point"
            " is %d.\n", result2);

    printf ("Temps total CUDA:"
            " %lf milseg.\n", time_taken2);
}
