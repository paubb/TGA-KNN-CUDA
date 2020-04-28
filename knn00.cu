#include <stdio.h>
#include <stdlib.h>

#define PINNED 0
#define THREADS 8

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

int main(int argc, char** argv)
{
    unsigned int N;
    unsigned int numBytes;
    unsigned int nBlocks, nThreads;
    
    float TiempoTotal;
    cudaEvent_t E0, E3;
    
    float * ref_points_dev   = NULL;
    float * dist_points_dev  = NULL;
    
    float * ref_points_host   = NULL;
    float * dist_points_host  = NULL;
    
    // numero de Threads
    nThreads = THREADS;

    // numero de Blocks en cada dimension 
    nBlocks = (N+nThreads-1)/nThreads; 
  
    numBytes = nBlocks * nThreads * sizeof(float);
    
    cudaEventCreate(&E0);
    cudaEventCreate(&E3);
    
    if (PINNED) {
        // Obtiene Memoria [pinned] en el host
        cudaMallocHost((float**)&ref_points_host, numBytes); 
        cudaMallocHost((float**)&dist_points_host, numBytes); 
    }
    else {
        // Obtener Memoria en el host
        ref_points_host = (float*) malloc(numBytes); 
        dist_points_host = (float*) malloc(numBytes);  
    }
    
    // Obtener Memoria en el device
    cudaMalloc((float**)&ref_points_dev, numBytes); 
    cudaMalloc((float**)&dist_points_dev, numBytes); 
    
    cudaEventRecord(E0, 0);
    
    // Copiar datos desde el host en el device 
    cudaMemcpy(ref_points_dev, ref_points_host, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dist_points_dev, dist_points_host,numBytes, cudaMemcpyHostToDevice);
    
    // Ejecutar el kernel 
}


