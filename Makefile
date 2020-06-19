CUDA_HOME   = /Soft/cuda/9.0.176

NVCC        = $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS  = -O3 -Wno-deprecated-gpu-targets -I$(CUDA_HOME)/include --ptxas-options=-v -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc
LD_FLAGS    = -Wno-deprecated-gpu-targets -lcudart -Xlinker -rpath,$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib

EXE00	        = knn00.exe
EXE01	        = knn01.exe
EXE02	        = knn02.exe
EXE03	        = knn03.exe
OBJ00	        = knn00.o
OBJ01	        = knn01.o
OBJ02	        = knn02.o
OBJ03	        = knn03.o

default: $(EXE03)

knn00.o: knn00.cu
	$(NVCC) -c -o $@ knn00.cu $(NVCC_FLAGS)

knn01.o: knn01.cu
	$(NVCC) -c -o $@ knn01.cu $(NVCC_FLAGS)

knn02.o: knn02.cu
	$(NVCC) -c -o $@ knn02.cu $(NVCC_FLAGS)

knn03.o: knn03.cu
	$(NVCC) -c -o $@ knn03.cu $(NVCC_FLAGS)

$(EXE00): $(OBJ00)
	$(NVCC) $(OBJ00) -o $(EXE00) $(LD_FLAGS)

$(EXE01): $(OBJ01)
	$(NVCC) $(OBJ01) -o $(EXE01) $(LD_FLAGS)

$(EXE02): $(OBJ02)
	$(NVCC) $(OBJ02) -o $(EXE02) $(LD_FLAGS)

$(EXE03): $(OBJ03)
	$(NVCC) $(OBJ03) -o $(EXE03) $(LD_FLAGS)

all:	$(EXE00) $(EXE01) $(EXE02) $(EXE03)

clean:
	rm -rf *.o *.exe
