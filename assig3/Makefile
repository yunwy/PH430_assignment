NVCC := nvcc
CXX := g++
#ARCH_FLAGS := -arch=sm_70
ARCH_FLAGS := -arch=sm_86 
MATH_FLAGS := --fmad=false -prec-div=true -prec-sqrt=true -ftz=false
CXXFLAGS := -std=c++20 -O3 -I$(HOME)/local/include 
LDFLAGS := -L$(HOME)/local/lib -lcnpy -lz

TARGET := dynamics

SRCS := main.cu
OBJS := $(SRCS:.cu=.o)

all: $(TARGET)

%.o: %.cu
	$(NVCC) $(ARCH_FLAGS) $(MATH_FLAGS) $(CXXFLAGS) -c $< -o $@ 

$(TARGET): $(OBJS)
	$(NVCC) $(ARCH_FLAGS) $(OBJS) -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET) $(OBJS)

.PHONY: all clean
