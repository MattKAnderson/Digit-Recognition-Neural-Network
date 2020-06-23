
CXX = g++
HEADER = include/neuralNet.h include/mnist_data.h
CFLAGS = -c -g 
INC = -Iinclude -I${MKLROOT}/include
MKL = -L${MKLROOT}/lib/intel64 -lmkl_rt -lpthread -lm -ldl 
SPEED = -O3 -march=native -DNDEBUG -ffast-math 
OBJ = neuralNet.o mnist_data.o main.o
VPATH = src

net: $(OBJ)
	$(CXX) $(SPEED) -o $@ $^ $(MKL)

#neuralNet.o: src/neuralNet.cpp $(HEADER)
#	$(CXX) $(INC) $(CFLAGS) $< -o $@
%.o: %.cpp $(HEADER)
	$(CXX) $(INC) $(CFLAGS) $(SPEED) $< -o $@ 

clean:
	rm *.o net

