#include <stdio.h>
#include <stdlib.h>

#include <math.h>
#include <iostream>     // std::cout
#include <algorithm>    // std::sort
#include <vector>       // std::vector
#include <time.h>
using namespace std;

#define PINNED 1
#define THREADS 1000

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

void InitHostFreq(unsigned int *freq1_host, unsigned int *freq2_host) {

    freq1_host[0] = 0;
    freq2_host[0] = 0;

}

__global__ void calculateDistance(int n, Point p, float *ref_points_dev_x, float *ref_points_dev_y, float *result_prediction_dev) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Fill distances of all points from p

        result_prediction_dev[i] =
            sqrt((ref_points_dev_x[i] - p.x) * (ref_points_dev_x[i] - p.x) +
                 (ref_points_dev_y[i] - p.y) * (ref_points_dev_y[i] - p.y));


}

__global__ void calculateFreq(int k, float *ref_points_host_val, unsigned int *freq1_dev, unsigned int *freq2_dev) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < k) {
        if (ref_points_host_val[i] == 0) {
            atomicAdd(&freq1_dev[0], 1);
        }
        else if (ref_points_host_val[i] == 1) {
            atomicAdd(&freq2_dev[0], 1);

        }
    }
}

int classifyAPointCUDA(Point arr[], float val[], int n, int k, Point p)
{
    unsigned int N;
    unsigned int numBytes;
    unsigned int nBlocks, nThreads;

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

    unsigned int *freq1_dev = NULL;
    unsigned int *freq2_dev = NULL;
    unsigned int *freq1_host = NULL;
    unsigned int *freq2_host = NULL;


    // numero de Threads
    nThreads = THREADS;

    // numero de Blocks en cada dimension
    nBlocks = (n+nThreads-1)/nThreads;
    printf("nBlocks = %d \n", nBlocks);

    numBytes = nBlocks * nThreads * sizeof(float);
    printf("numBytes = %d \n", numBytes);

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
        cudaMallocHost((float**)&ref_points_host_val, numBytes);
        cudaMallocHost((float**)&result_prediction_host, numBytes);

        cudaMallocHost((float**)&freq1_host, sizeof(unsigned int));
        cudaMallocHost((float**)&freq2_host, sizeof(unsigned int));

    } else {
        // Obtener Memoria en el host
        ref_points_host_x = (float*) malloc(numBytes);
        ref_points_host_y = (float*) malloc(numBytes);
        ref_points_host_val = (float*) malloc(numBytes);
        result_prediction_host = (float*) malloc(numBytes);

        freq1_host = (unsigned int*) malloc(sizeof(unsigned int));
        freq2_host = (unsigned int*) malloc(sizeof(unsigned int));
    }

    InitHostInput(arr, n, val, p, ref_points_host_x, ref_points_host_y, ref_points_host_val);

    InitHostFreq(freq1_host, freq2_host);

    // Obtener Memoria en el device
    cudaMalloc((float**)&ref_points_dev_x, numBytes);
    cudaMalloc((float**)&ref_points_dev_y, numBytes);
    cudaMalloc((float**)&ref_points_dev_val, numBytes);
    cudaMalloc((float**)&result_prediction_dev, numBytes);

    cudaMalloc((unsigned int**)&freq1_dev, sizeof(unsigned int));
    cudaMalloc((unsigned int**)&freq2_dev, sizeof(unsigned int));


    // Copiar datos desde el host en el device
    cudaMemcpy(ref_points_dev_x, ref_points_host_x, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(ref_points_dev_y, ref_points_host_y, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(ref_points_dev_val, ref_points_host_val, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(result_prediction_dev, result_prediction_host,numBytes, cudaMemcpyHostToDevice);

    cudaMemcpy(freq1_dev, freq1_host, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(freq2_dev, freq2_host, sizeof(unsigned int), cudaMemcpyHostToDevice);

    nBlocks = nBlocks-1;


    cudaEventRecord(E0, 0);

    // Ejecutar el kernel
    calculateDistance<<<nBlocks, nThreads>>>(n, p, ref_points_dev_x, ref_points_dev_y, result_prediction_dev);

    cudaEventRecord(E1, 0); cudaEventSynchronize(E1);
    cudaEventElapsedTime(&TiempoKernelDistance,  E0, E1);

    // Obtener el resultado desde el host
    cudaMemcpy(result_prediction_host, result_prediction_dev, numBytes, cudaMemcpyDeviceToHost);

    // Liberar Memoria del device
    cudaFree(ref_points_dev_x);
    cudaFree(ref_points_dev_y);
    cudaFree(result_prediction_dev);

    cudaEventRecord(E4, 0);
    // Sort the Points by distance from p
    quickSort(result_prediction_host, ref_points_host_val, 0, n-1);
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
    calculateFreq<<<k, 1>>>(k, ref_points_dev_val, freq1_dev, freq2_dev);

    cudaEventRecord(E3, 0); cudaEventSynchronize(E3);
    cudaEventElapsedTime(&TiempoKernelFreq,  E2, E3);

    TiempoAllOperations = TiempoKernelDistance + TiempoSort + TiempoKernelFreq;

    cudaMemcpy(freq1_host, freq1_dev, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(freq2_host, freq2_dev, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaFree(ref_points_dev_val);
    cudaFree(freq1_dev);
    cudaFree(freq2_dev);

    int result = -1;
    if(freq1_host[0] > freq2_host[0]) result = 0;
    else result = 1;

    printf ("freq1 is %d.\n", freq1_host[0]);
    printf ("freq2 is %d.\n", freq2_host[0]);

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
        cudaFreeHost(ref_points_host_x); cudaFreeHost(ref_points_host_y); cudaFreeHost(ref_points_host_val);cudaFreeHost(result_prediction_host); cudaFreeHost(freq1_host); cudaFreeHost(freq2_host);
    } else {
        free(ref_points_host_x); free(ref_points_host_y); free(ref_points_host_val); free(result_prediction_host); free(freq1_host); free(freq2_host);
    }

    cudaDeviceReset();

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
    n = 100000; // Number of data points
    Point arr[n];

    float val[n];

    for(int i = 0; i < n; ++i) {
        arr[i].x = rand() % 100;
        arr[i].y = rand() % 100;
        val[i] = rand() % 2;
    }

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
    int result = classifyAPoint(arr, n, k, p, val);
    t = clock() - t;
    float time_taken = ((float)t)/(CLOCKS_PER_SEC/1000); // in seconds

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
    int result2 = classifyAPointCUDA(arr,val, n, k, p);
    t2 = clock() - t2;
    float time_taken2 = ((float)t2)/CLOCKS_PER_SEC; // in seconds

    printf ("The value classified to unknown point"
            " is %d.\n", result2);

    printf ("Temps total CUDA:"
            " %lf seg.\n", time_taken2);
}
