#!/bin/bash
export PATH=/Soft/cuda/9.0.176/bin:$PATH

### Directivas para el gestor de colas
# Asegurar que el job se ejecuta en el directorio actual
#$ -cwd
# Asegurar que el job mantiene las variables de entorno del shell lamador
#$ -V
# Cambiar el nombre del job
#$ -N KNN 
# Cambiar el shell
#$ -S /bin/bash


./knn.exe

nvprof --unified-memory-profiling off --print-gpu-summary ./knn.exe
nvprof --unified-memory-profiling off --print-gpu-trace ./knn.exe
nvprof --unified-memory-profiling off --metrics
