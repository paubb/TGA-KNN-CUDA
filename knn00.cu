#include <stdio.h>
#include <stdlib.h>

#include <math.h>
#include <iostream>     // std::cout
#include <algorithm>    // std::sort
#include <vector>       // std::vector
#include <time.h>
using namespace std;

#define PINNED 0
#define THREADS 100

struct Point
{
    int val;     // Group of point
    double x, y;     // Co-ordinate of point
    double distance; // Distance from test point
};

// Used to sort an array of points by increasing
// order of distance
bool comparison(Point a, Point b)
{
    return (a.distance < b.distance);
}

// Used to sort an array of points by increasing
// order of distance
bool comparisonNOPoints(double a, double b)
{
    return (a < b);
}

void selectionSort(double *result_prediction_host, double *ref_points_host_val, int n) {
   int i, j, min, temp, temp2;
   for (i = 0; i < n - 1; i++) {
      min = i;
      for (j = i + 1; j < n; j++)
         if (result_prediction_host[j] < result_prediction_host[min])
            min = j;
      temp = result_prediction_host[i];
      temp2 = ref_points_host_val[i];
      result_prediction_host[i] = result_prediction_host[min];
      ref_points_host_val[i] = ref_points_host_val[min];
      result_prediction_host[min] = temp;
      ref_points_host_val[min] = temp2;
   }
}

/**
 * @param arr    refence points
 * @param n      number of reference points
 * @param k      number of points we want to use for the prediction
 * @param p      point we want to predict
 */
int classifyAPoint(Point arr[], int n, int k, Point p)
{
    // Fill distances of all points from p
    for (int i = 0; i < n; i++)
        arr[i].distance =
            sqrt((arr[i].x - p.x) * (arr[i].x - p.x) +
                 (arr[i].y - p.y) * (arr[i].y - p.y));

    // Sort the Points by distance from p
    sort(arr, arr+n, comparison);

    // Now consider the first k elements and only
    // two groups
    int freq1 = 0;     // Frequency of group 0
    int freq2 = 0;     // Frequency of group 1
    for (int i = 0; i < k; i++)
    {
        if (arr[i].val == 0)
            freq1++;
        else if (arr[i].val == 1)
            freq2++;
    }

    return (freq1 > freq2 ? 0 : 1);
}

void InitHostInput(Point arr[], int n, Point p, double *ref_points_host_x, double *ref_points_host_y, double *ref_points_host_val) {

    for (int i=0; i<n; i++) {
        ref_points_host_x[i] = arr[i].x;
        ref_points_host_y[i] = arr[i].y;
        ref_points_host_val[i] = arr[i].val;
    }

}

__global__ void calculateDistance(int n, Point p, double *ref_points_dev_x, double *ref_points_dev_y, double *result_prediction_dev) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Fill distances of all points from p
    if(i < n) {
        result_prediction_dev[i] =
            sqrt((ref_points_dev_x[i] - p.x) * (ref_points_dev_x[i] - p.x) +
                 (ref_points_dev_y[i] - p.y) * (ref_points_dev_y[i] - p.y));
    }

}

__global__ void calculateFreq(int k, double *ref_points_host_val, double *result_freq_dev) {


    // Now consider the first k elements and only
    // two groups
    double freq1 = 0;     // Frequency of group 0
    double freq2 = 0;     // Frequency of group 1
    for (int i = 0; i < k; i++)
    {
        if (ref_points_host_val[i] == 0)
            freq1++;
        else if (ref_points_host_val[i] == 1)
            freq2++;

    }
    if(freq1 > freq2) result_freq_dev[0] = 0.0f;
    else result_freq_dev[0] = 1.0f;
}

int classifyAPointCUDA(Point arr[], int n, int k, Point p)
{
    unsigned int N;
    unsigned int numBytes;
    unsigned int nBlocks, nThreads;

    double TiempoTotal;
    cudaEvent_t E0, E3;

    double *ref_points_dev_x   = NULL;
    double *ref_points_dev_y   = NULL;
    double *ref_points_dev_val   = NULL;
    double *result_prediction_dev  = NULL;

    double *result_freq_dev = NULL;

    double *result_freq_host = NULL;

    double *ref_points_host_x   = NULL;
    double *ref_points_host_y = NULL;
    double *ref_points_host_val   = NULL;
    double *result_prediction_host  = NULL;

    // numero de Threads
    nThreads = THREADS;

    // numero de Blocks en cada dimension
    nBlocks = (n+nThreads-1)/nThreads;
    printf("nBlocks = %d \n", nBlocks);

    numBytes = nBlocks * nThreads * sizeof(double);
    printf("numBytes = %d \n", numBytes);

    cudaEventCreate(&E0);
    cudaEventCreate(&E3);

    // Obtener Memoria en el host
    ref_points_host_x = (double*) malloc(numBytes);
    ref_points_host_y = (double*) malloc(numBytes);
    ref_points_host_val = (double*) malloc(numBytes);
    result_prediction_host = (double*) malloc(numBytes);

    result_freq_host = (double*) malloc(numBytes);

    InitHostInput(arr, n, p, ref_points_host_x, ref_points_host_y, ref_points_host_val);

    // Obtener Memoria en el device
    cudaMalloc((double**)&ref_points_dev_x, numBytes);
    cudaMalloc((double**)&ref_points_dev_y, numBytes);
    cudaMalloc((double**)&ref_points_dev_val, numBytes);
    cudaMalloc((double**)&result_prediction_dev, numBytes);

    cudaMalloc((double**)&result_freq_dev, numBytes);


    // Copiar datos desde el host en el device
    cudaMemcpy(ref_points_dev_x, ref_points_host_x, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(ref_points_dev_y, ref_points_host_y, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(ref_points_dev_val, ref_points_host_val, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(result_prediction_dev, result_prediction_host,numBytes, cudaMemcpyHostToDevice);

    cudaMemcpy(result_freq_dev, result_freq_host, numBytes, cudaMemcpyHostToDevice);

    cudaEventRecord(E0, 0);

    // Ejecutar el kernel
    calculateDistance<<<nBlocks, nThreads>>>(n, p, ref_points_dev_x, ref_points_dev_y, result_prediction_dev);

    cudaEventRecord(E3, 0); cudaEventSynchronize(E3);
    cudaEventElapsedTime(&TiempoTotal,  E0, E3);

    // Obtener el resultado desde el host
    cudaMemcpy(result_prediction_host, result_prediction_dev, numBytes, cudaMemcpyDeviceToHost);

    // Liberar Memoria del device
    cudaFree(ref_points_dev_x);
    cudaFree(ref_points_dev_y);
    cudaFree(result_prediction_dev);


    // Sort the Points by distance from p
    selectionSort(result_prediction_host, ref_points_host_val, n);
    /*
        for(int i = 0; i < n; i++){
        printf("L'element: %d\n", i);
        printf("La distancia: %f\n", result_prediction_host[i]);
        printf("La x: %f\n", ref_points_host_val[i]);
    }
    */

    // Ejecutar el kernel
    calculateFreq<<<nBlocks, nThreads>>>(k, ref_points_dev_val, result_freq_dev);

    cudaMemcpy(result_freq_host, result_freq_dev, numBytes, cudaMemcpyDeviceToHost);

    cudaFree(ref_points_dev_val);
    cudaFree(result_freq_dev);

    printf ("The value classified to unknown point"
            " is %f.\n", result_freq_host[0]);
    double result = result_freq_host[0];

    printf("Invocació Kernel <<<nBlocks, nKernels>>> (N): <<<%d, %d>>> (%d)\n", nBlocks, nThreads, n);

    printf("Tiempo Global (00): %4.6f milseg\n", TiempoTotal);

    free(ref_points_host_x); free(ref_points_host_y); free(ref_points_host_val); free(result_prediction_host); free(result_freq_host);

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

    //Es declaren les variables
    int n, k;
    struct Point p;

    //S'inicialitza la K, i les coordenades del Testing point
    if (argc == 1)      { InitDefecte(&k, &p); }
    else if (argc == 2) { k = atoi(argv[1]); InitTestPointDefecte(&p); }
    else if (argc == 4) { k = atoi(argv[1]); p.x = atof(argv[2]); p.y = atof(argv[3]);}
    else { printf("Usage: ./exe k TestPointCoordenadaX TestPointCoordenadaY\n"); exit(0); }

    //Es crea l'estructura sobre la qual es vol fer la predicció
    n = 1000; // Number of data points
    Point arr[n];

    for(int i = 0; i < n; ++i) {
        arr[i].x = rand() % 100;
        arr[i].y = rand() % 100;
        arr[i].val = rand() % 2;
    }

    printf("k = %d \n", k);

    printf("The Testing Point values are:");
    printf(" x = %f", p.x);
    printf(" and");
    printf(" y = %f", p.y);
    printf("\n");

    // Calculate the time taken by the sequential code: classifyAPoint function
    clock_t t;
    t = clock();
    int result = classifyAPoint(arr, n, k, p);
    t = clock() - t;
    float time_taken = ((float)t)/CLOCKS_PER_SEC; // in seconds

    printf ("The value classified to unknown point"
            " is %d.\n", result);

    printf ("Temps seqüencial:"
            " is %lf.\n", time_taken);

    printf("---------------------------------------------------------------------- \n");

    // Calculate the time taken by the sequential code: classifyAPoint function
    clock_t t2;
    t2 = clock();
    int result2 = classifyAPointCUDA(arr, n, k, p);
    t2 = clock() - t2;
    float time_taken2 = ((float)t2)/CLOCKS_PER_SEC; // in seconds

    printf ("The value classified to unknown point"
            " is %d.\n", result2);

    printf ("Temps CUDA:"
            " is %lf.\n", time_taken2);

}
