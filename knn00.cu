#include <stdio.h>
#include <stdlib.h>

#include <math.h> 
#include <iostream>     // std::cout
#include <algorithm>    // std::sort
#include <vector>       // std::vector
#include <time.h> 
using namespace std;

#define PINNED 0
#define THREADS 1000

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

void InitHostInput(Point arr[], int n, Point p, float *ref_points_host_x, float *ref_points_host_y, float *result_prediction_host) {

    for (int i=0; i<n; i++) 
        ref_points_host_x[i] = arr[i].x;
        ref_points_host_y[i] = arr[i].y;
    
}

__global__ void calculateDistance(int n, Point p, float *ref_points_dev_x, float *ref_points_dev_y, float *result_prediction_dev) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Fill distances of all points from p 
    if(i < n) { 
        result_prediction_dev[i] = 
            sqrt((ref_points_dev_x[i] - p.x) * (ref_points_dev_x[i] - p.x) + 
                 (ref_points_dev_y[i] - p.y) * (ref_points_dev_y[i] - p.y)); 
    }
    
}

int classifyAPointCUDA(Point arr[], int n, int k, Point p) 
{ 
    unsigned int N;
    unsigned int numBytes;
    unsigned int nBlocks, nThreads;
    
    float TiempoTotal;
    cudaEvent_t E0, E3;
    
    float * ref_points_dev_x   = NULL;
    float * ref_points_dev_y   = NULL;
    float * result_prediction_dev  = NULL;
    
    float * ref_points_host_x   = NULL;
    float * ref_points_host_y = NULL;
    float * result_prediction_host  = NULL;
    
    // numero de Threads
    nThreads = THREADS;

    // numero de Blocks en cada dimension 
    nBlocks = (n+nThreads-1)/nThreads; 
    printf("nBlocks = %d \n", nBlocks);
    
    numBytes = nBlocks * nThreads * sizeof(float);
    printf("numBytes = %d \n", numBytes);
    
    cudaEventCreate(&E0);
    cudaEventCreate(&E3);
    
    // Obtener Memoria en el host
    ref_points_host_x = (float*) malloc(numBytes); 
    ref_points_host_y = (float*) malloc(numBytes); 
    result_prediction_host = (float*) malloc(numBytes);  
    
    InitHostInput(arr[], n, p, ref_points_host_x, ref_points_host_y, result_prediction_host);
    
    // Obtener Memoria en el device
    cudaMalloc((float**)&ref_points_dev_x, numBytes); 
    cudaMalloc((float**)&ref_points_dev_y, numBytes); 
    cudaMalloc((float**)&result_prediction_dev, numBytes); 
    
    cudaEventRecord(E0, 0);
    
    // Copiar datos desde el host en el device 
    cudaMemcpy(ref_points_dev_x, ref_points_host_x, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(ref_points_dev_y, ref_points_host_y, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(result_prediction_dev, result_prediction_host,numBytes, cudaMemcpyHostToDevice);
    
    // Ejecutar el kernel 
    calculateDistance<<<nBlocks, nThreads>>>(n, p, ref_points_dev_x, ref_points_dev_y, result_prediction_dev);
    
    // Obtener el resultado desde el host 
    cudaMemcpy(result_prediction_host, result_prediction_dev, numBytes, cudaMemcpyDeviceToHost);
    
    cudaEventRecord(E3, 0); cudaEventSynchronize(E3);
    
    // Liberar Memoria del device 
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    cudaEventElapsedTime(&TiempoTotal,  E0, E3);
    printf("Invocació Kernel <<<nBlocks, nKernels>>> (N): <<<%d, %d>>> (%d)\n", nBlocks, nThreads, n);
    
    printf("Tiempo Global (00): %4.6f milseg\n", TiempoTotal);
        
    free(ref_points_host_x); free(ref_points_host_y); free(result_prediction_host);
      
    return 0;
    
}

void InitKDefecte(int *k) {
    
    // Parameter to decide group of the testing point 
    (*k) = 3;
    
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
    n = 1000000; // Number of data points 
    Point arr[n];
    
    for(int i = 0; i < n; ++i) {
        arr[i].x = rand() % 100; 
        arr[i].y = rand() % 100; 
        arr[i].val = rand() % 2;
    }
    
    printf("k = %d \n", k);
    
    printf("The Testing Point values are:");
    printf(" x = %lf", p.x);
    printf(" and");
    printf(" y = %lf", p.y);
    printf("\n");
    
    // Calculate the time taken by the sequential code: classifyAPoint function 
    clock_t t; 
    t = clock(); 
    int result = classifyAPoint(arr, n, k, p); 
    t = clock() - t; 
    double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds 
    
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
    double time_taken2 = ((double)t2)/CLOCKS_PER_SEC; // in seconds 
    
    printf ("The value classified to unknown point"
            " is %d.\n", result2);
            
    printf ("Temps CUDA:"
            " is %lf.\n", time_taken2);
    
}


