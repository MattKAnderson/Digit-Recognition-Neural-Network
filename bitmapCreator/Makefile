CPP = g++
INC = -I../include
OBJ = main.o bitmapCreator.o mnist_data.o

.PHONY: clean

imgTest: $(OBJ)
	$(CPP) -o $@ $(INC) $^

%.o: %.cpp
	$(CPP) -c -o $@ $(INC) $<

mnist_data.o: ../src/mnist_data.cpp 
	$(CPP) -c -o $@ $(INC) $<

clean:
	rm *.o
