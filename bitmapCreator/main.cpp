#include <iostream>
#include <string>
#include "mnist_data.h"
#include "bitmapCreator.h"

int main() {
	std::string imageDirectory = "./testImages/";
	std::string trainImgLoc = "../imgData/train-images.idx3-ubyte";
	std::string trainlblLoc = "../imgData/train-labels.idx1-ubyte";
	mnist_data trainData(trainImgLoc, trainlblLoc);
	bitmapCreator builder;

	// build the first 5 images
	for (int i = 0; i < 5; i++) {
		mnist_data::image tmp = trainData.imgAt(i);
		std::vector<int> img(tmp.pixelIntensity.size());
		for (int j = 0; j < img.size(); j++) { 
			img[j] = static_cast<int>(tmp.pixelIntensity[j] * 255.0); 
		}
		int label = trainData.labelAt(i);
		std::string fileName = "sample_" + std::to_string(i) + "_label_"
								+ std::to_string(label) + ".bmp";
		std::cout << "right before write bitmap" << std::endl;
		builder.writeBitmap(imageDirectory + fileName, img, 28); 
	}

}
