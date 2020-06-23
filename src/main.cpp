
#include <chrono>
#include <iostream>
#include <string>
#include "neuralNet.h"
#include "mnist_data.h"


int main() {
	long long unsigned hiddenLayerSize = 800;
	std::string trainImgLoc = "imgData/train-images.idx3-ubyte";
	std::string trainlblLoc = "imgData/train-labels.idx1-ubyte";
	std::string testImgLoc = "imgData/t10k-images.idx3-ubyte";
	std::string testlblLoc = "imgData/t10k-labels.idx1-ubyte";

	//creating test and training data objects
	mnist_data trainData(trainImgLoc, trainlblLoc);
	mnist_data testData(testImgLoc, testlblLoc);
	std::cout << "Created data_set objects.." << std::endl;
	
	//declaring network
	neuralNet network({784, hiddenLayerSize, 10});
	std::cout << "Initialized neural network.." << std::endl;

	//testing
	std::cout << "Beginning timer....\n";
	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
	std::cout << "Beginning Stochastic gradient descent.." << std::endl;
	network.SGD(trainData, 500, 50, 3.0, 0.1, &testData);
	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
	unsigned long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "elapsed time for computation is: " << static_cast<double>(ms) / 1000.0 << " s\n";
}
