#!/bin/bash
export PATH=/Soft/cuda/9.0.176/bin:$PATH

### Directivas para el gestor de colas
# Asegurar que el job se ejecuta en el directorio actual
#$ -cwd
# Asegurar que el job mantiene las variables de entorno del shell lamador
#$ -V
# Cambiar el nombre del job
#$ -N STREAMS 
# Cambiar el shell
#$ -S /bin/bash


./knn.exe
#nvprof --print-gpu-summary --unified-memory-profiling off ./knn.exe

