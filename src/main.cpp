

#include <iostream>
#include <string>
#include "neuralNet.h"
#include "mnist_data.h"


int main() {
	std::string trainImgLoc = "imgData/train-images.idx3-ubyte";
	std::string trainlblLoc = "imgData/train-labels.idx1-ubyte";
	std::string testImgLoc = "imgData/t10k-images.idx3-ubyte";
	std::string testlblLoc = "imgData/t10k-labels.idx1-ubyte";

	//creating test and training data objects
	mnist_data trainData(trainImgLoc, trainlblLoc);
	mnist_data testData(testImgLoc, testlblLoc);
	std::cout << "Created data_set objects.." << std::endl;
	
	//declaring network
	neuralNet network({784, 100, 10});
	std::cout << "Initialized neural network.." << std::endl;

	//testing
	std::cout << "Beginning Stochastic gradient descent.." << std::endl;
	network.SGD(trainData, 30, 10, 3.0, &testData);


}
