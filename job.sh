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


#./knn00.exe
#nvprof --unified-memory-profiling off --print-gpu-summary ./knn00.exe
nvprof --unified-memory-profiling off --print-gpu-trace ./knn00.exe

#./knn01.exe
#nvprof --unified-memory-profiling off --print-gpu-summary ./knn01.exe
#nvprof --unified-memory-profiling off --print-gpu-trace ./knn01.exe

#./knn02.exe
#nvprof --unified-memory-profiling off --print-gpu-summary ./knn02.exe
#nvprof --unified-memory-profiling off --print-gpu-trace ./knn02.exe

#./knn03.exe
#nvprof --unified-memory-profiling off --print-gpu-summary ./knn03.exe
#nvprof --unified-memory-profiling off --print-gpu-trace ./knn03.exe

#./knn04.exe
#nvprof --unified-memory-profiling off --print-gpu-summary ./knn04.exe
#nvprof --unified-memory-profiling off --print-gpu-trace ./knn04.exe

