#include "mnist_data.h"
#include <iostream>
//validate the image files and load into memory the image data
mnist_data::mnist_data(std::string imgLoc, std::string labelLoc) {
    char fourBytes[4];
    int readInt;
    int nItems;
    int rows;
    int cols;
    int pixelsPerImage;
    std::ifstream imgFile(imgLoc, std::ios::binary);
    std::ifstream lblFile(labelLoc, std::ios::binary);

    //file validations
    imgFile.read(fourBytes, 4);
    readInt = charArrayToInt(fourBytes);
    if (readInt != 2051) { throw "Magic number for image file incorrect!"; }
    lblFile.read(fourBytes, 4);
    readInt = charArrayToInt(fourBytes);
    if (readInt != 2049) { throw "Magic number for label file incorrect!"; }
    imgFile.read(fourBytes, 4);
    nItems = charArrayToInt(fourBytes);
    lblFile.read(fourBytes, 4);
    readInt = charArrayToInt(fourBytes);
    if (nItems != readInt) {
        throw "Number of items in each file do not match!"; 
    }
    //reading data array sizes, resizing, and reading image data
    imgFile.read(fourBytes, 4);
    rows = charArrayToInt(fourBytes);
    imgFile.read(fourBytes, 4);
    cols = charArrayToInt(fourBytes);
    pixelsPerImage = cols * rows;
    imgLabelPairs.resize(nItems);
    for (auto it = imgLabelPairs.begin(); it != imgLabelPairs.end(); it++) {
        it->first.pixelIntensity.resize(pixelsPerImage);
        lblFile.read(fourBytes, 1);
        it->second = static_cast< unsigned char >(fourBytes[0]);
        for (int i = 0; i < pixelsPerImage; i++) {
            imgFile.read(fourBytes, 1);
			unsigned temp = static_cast<unsigned>(static_cast<unsigned char>(fourBytes[0]));
            it->first.pixelIntensity[i] = static_cast< double >(temp);
            it->first.pixelIntensity[i] /= 255.0;
        }
    }
}

//Getter methods
int mnist_data::numImages() const { return imgLabelPairs.size(); }
int mnist_data::labelAt(int ID) const { return imgLabelPairs[ID].second; }
const mnist_data::image& mnist_data::imgAt(int ID) const { return imgLabelPairs[ID].first; }

//shuffle the images (with their label) into a random order
void mnist_data::shuffle() {
    using namespace std::chrono;
    typedef high_resolution_clock clk;
    uint64_t seed;
    seed = duration_cast< microseconds >(clk::now().time_since_epoch()).count();
    std::mt19937 rng(seed);
    std::shuffle(imgLabelPairs.begin(), imgLabelPairs.end(), rng);
}

int mnist_data::charArrayToInt(char* bytes) {
    int invert;
    invert = static_cast< int >(static_cast< unsigned char >(bytes[0]) << 24 |
                                static_cast< unsigned char >(bytes[1]) << 16 |
                                static_cast< unsigned char >(bytes[2]) << 8 |
                                static_cast< unsigned char >(bytes[3]));
	return invert;
}



















