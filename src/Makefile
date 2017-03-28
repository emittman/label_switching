NVCC_FLAGS = -arch=sm_35 -Xcompiler -fPIC

DEBUG_FLAGS = -DTHRUST_DEBUG

R_INC = ${R_HOME}/include

BOOST_INC = /usr/include/boost

CUDA_PATH = /usr/local/cuda-8.0/lib64

LIBS = -lcublas_device -lcublas -lcurand -lcuda -lcudart

TARGET = label_switch.so

TEST_OBJS = test.o

#final SHLIB
$(TARGET): $(TEST_OBJS)
	nvcc -shared $(NVCC_FLAGS) $(TEST_OBJS) -o $(TARGET) -I$(R_INC)\
-L$(CUDA_PATH) $(LIBS) $(DL_FLAGS) $(DEBUG_FLAGS)

test.o: test.cu
	nvcc $(NVCC_FLAGS) -I$(R_INC) -c test.cu -o test.o $(LIBS)
