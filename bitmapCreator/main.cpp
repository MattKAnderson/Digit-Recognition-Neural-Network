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
		std::string filename = "sample_" + std::to_string(i) + "_label_"
								+ std::to_string(label) + ".bmp";
		std::cout << "right before write bitmap" << std::endl;
		builder.writeBitmap(imageDirectory + filename, img, 28); 
	}

	// test image rotation
	mnist_data::image orig = trainData.imgAt(5);
	std::vector<int> img(orig.pixelIntensity.size());
	for (int i = 0; i < 10; i++) {
		mnist_data::image rot = mnist_data::distortImage(orig, 0, 0, 0, 0, i);
		std::vector<int> rotImg(rot.pixelIntensity.size());
		for (int j = 0; j < img.size(); j++) { 
			img[j] = static_cast<int>(orig.pixelIntensity[j] * 255.0); 
			rotImg[j] = static_cast<int>(rot.pixelIntensity[j] * 255.0); 
		}
		builder.writeBitmap(imageDirectory + "rotationTest" + std::to_string(i) + ".bmp", rotImg, 28); 
	}
	builder.writeBitmap(imageDirectory + "rotationTestOriginal.bmp", img, 28); 
}
