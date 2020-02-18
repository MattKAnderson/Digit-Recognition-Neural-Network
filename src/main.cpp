

#include <iostream>
#include <string>
#include "neuralNet.h"
#include "mnist_data.h"


int main() {
	std::string trainImgLoc = "../imgData/train-images.idx3-ubyte";
	std::string trainlblLoc = "../imgData/train-labels.idx1-ubyte";
	std::string testImgLoc = "../imgData/t10k-images.idx3-ubyte";
	std::string testlblLoc = "../imgData/t10k-labels.idx1-ubyte";

	//creating test and training data objects
	mnist_data trainData(trainImgLoc, trainlblLoc);
	mnist_data testData(testImgLoc, testlblLoc);

	//declaring network
	neuralNet network({784, 30, 10});

	//testing
	network.SGD(trainData, 20, 100, 3.0, &testData);

}
