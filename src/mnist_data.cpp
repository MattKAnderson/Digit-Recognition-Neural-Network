#inlcude "../include/mnist_data.h"

//validate the image files and load into memory the image data
mnist_data::mnist_data(std::string imgLoc, std::string labelLoc) {
    char fourBytes[4];
    int readInt;
    int nItems;
    int rows;
    int cols;
    int pixelPerImage;
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
    pixelPerImage = cols * rows;
    imgLabelPairs.resize(nItems);
    for (auto it = imgLabelPairs.begin(); it != imgLabelPairs.end(); it++) {
        it->first.pixelIntensity.resize(pixelsPerImage);
        lblFile.read(fourBytes, 1);
        it->second = static_cast< int >(fourBytes[0]);
        for (int i = 0; i < pixelsPerImage; i++) {
            imgFile.read(fourBytes, 1);
            it->first.pixelIntensity[i] = static_cast< float >(fourBytes[0]);
            it->first.pixelIntensity[i] /= 255.0;
        }
    }
}

//Getter methods
int mnist_data::numImages() { return imgLabelPairs.size(); }
int mnist_data::labelAt(int ID) { return imgLabelParis[ID].second; }
image& mnist_data::imgAt(int ID) { return imgLabelPairs[ID].first; }

//shuffle the images (with their label) into a random order
void mnist_data::shuffle() {
    using std::chrono;
    using std::chrono::high_resolution_clock;
    uint64_t seed;
    seed = duration_cast< microseconds >(now().time_since_epoch()).count();
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



















