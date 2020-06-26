
#define _USE_MATH_DEFINES

#include "mnist_data.h"
#include <iostream>
#include <cmath>
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

void mnist_data::expandWithDistortions(int factor, 
									   double rotationFactor,
									   uint64_t seed)
{
	std::vector<std::pair<image, int>> newSet(imgLabelPairs.size() * factor);
	std::mt19937 gen(seed);
	int pos = 0;
	for (std::pair<image, int> pair : imgLabelPairs) {
		newSet[pos++] = pair;
		for (int i = 0; i < factor; i++) {
			newSet[pos++] = std::make_pair(distortImage(pair.first, 10.0, gen), 
										   pair.second);
		}
	}
	imgLabelPairs = newSet;
}

/*
 *  ****need to split into multiple smaller functions for easier maintenance
 */
mnist_data::image mnist_data::distortImage(const mnist_data::image& original,
										   double rotationFactor,
										   std::mt19937& gen)
{
	std::normal_distribution<> dist(0, rotationFactor);

	// start with rotation -- fixed example
	double random = dist(gen);
	double radians = random * M_PI / 180.0;
	double cosine = std::cos(radians);
	double sine = std::sin(radians);
	std::cout << "random angle is: " << random << std::endl;
	/*
     *   Rotation matrix is 
	 *		[ cos(t)  -sin(t) ]
	 *      [ sin(t)   cos(t) ]
	 *
	 *   Using an interpolation scheme, points rotate to decimal points.
	 *   Each original pixel contributes fractionally to the intensity of
	 *   the four nearby points. The resulting intensity is the average of
	 *   all contributing points weighted by their euclidean distance to the
	 *   integer point of interest
	 */

	image newImg;
	newImg.pixelIntensity.resize(original.pixelIntensity.size(), -1);

	// create a list of all pixels adjacent to lattice points after rotation
	std::vector<std::vector<pixel>> 
	gridAdjacencyList(original.pixelIntensity.size());
	
	for (int i = 0; i < original.pixelIntensity.size(); i++) {
		
		// get coordinate from center of image for this pixel
		pixel p;
		p.x = static_cast<double>(i % 28) - 13.5;
		p.y = static_cast<double>(i / 28) - 13.5;
		p.intensity = original.pixelIntensity[i]; 

		// applying rotation
		double pxTemp = p.x;

		p.x = (p.x * cosine - p.y * sine) + 13.5;
		p.y = (pxTemp * sine + p.y * cosine) + 13.5;

		// add to all 4 points on edge of grid-square around pixel location
		int floorX = std::floor(p.x);
		int ceilX = std::ceil(p.x);
		int floorY = std::floor(p.y);
		int ceilY = std::ceil(p.y);

		// cases for how many grid points to add the pixel to, with bounds
		// checking
		if (floorX == ceilX) {
			if (floorY == ceilY) {
				if (floorX >= 0 && floorY >= 0 && floorX < 28 && floorY < 28)
				{
					gridAdjacencyList[floorX + 28 * floorY].push_back(p);
				}
			}
			else {
				if (floorX >= 0 && floorY >= 0 && floorX < 28 && floorY < 28)
				{
					gridAdjacencyList[floorX + 28 * floorY].push_back(p);
				}
				if (floorX >= 0 && ceilY >= 0 && floorX < 28 && ceilY < 28)
				{
					gridAdjacencyList[floorX + 28 * ceilY].push_back(p);
				}
			}
		}
		else if (floorY == ceilY) {
			if (floorX >= 0 && floorY >= 0 && floorX < 28 && floorY < 28) {
				gridAdjacencyList[floorX + 28 * floorY].push_back(p);
			}
			if (ceilX >= 0 && floorY >= 0 && ceilX < 28 && floorY < 28) {
				gridAdjacencyList[ceilX + 28 * floorY].push_back(p);
			}
		}
		else {
			if (floorX >= 0 && floorY >= 0 && floorX < 28 && floorY < 28) {
				gridAdjacencyList[floorX + 28 * floorY].push_back(p);
			}
			if (ceilX >= 0 && floorY >= 0 && ceilX < 28 && floorY < 28) {
				gridAdjacencyList[ceilX + 28 * floorY].push_back(p);
			}
			if (floorX >= 0 && ceilY >= 0 && floorX < 28 && ceilY < 28) {
				gridAdjacencyList[floorX + 28 * ceilY].push_back(p);
			}
			if (ceilX >= 0 && ceilY >= 0 && ceilX < 28 && ceilY < 28) {
				gridAdjacencyList[ceilX + 28 * ceilY].push_back(p);
			}
		}
	}

	// use grid adjacency list to interpolate pixel intensity values
	/*
     *   ** summing weighted squares and taking square roots, need to weight 
	 *      heigher intensities more
	 */
	double greatestIntensity = 0.0;
	for (int i = 0; i < gridAdjacencyList.size(); i++) {
		int targetX = i % 28;
		int targetY = i / 28;

		// iterate over adjacent points, use their euclidean distance divided
		// by sqrt(2) to weigh the contibution to new pixel
		newImg.pixelIntensity[i] = 0.0;
		int contributions = 0;
		for (pixel p : gridAdjacencyList[i]) {
			double dx = p.x - static_cast<double>(targetX);
			double dy = p.y - static_cast<double>(targetY);
			double weight = 1.0 - std::sqrt((dx * dx + dy * dy) / 2.0);
			if (weight < 0.0) { std::cout << "weight error!\n"; }
			newImg.pixelIntensity[i] += weight * p.intensity * p.intensity;
			contributions++;
		}
		newImg.pixelIntensity[i] /= static_cast<double>(contributions);
		newImg.pixelIntensity[i] = std::sqrt(newImg.pixelIntensity[i]);
		if (newImg.pixelIntensity[i] > greatestIntensity) {
			greatestIntensity = newImg.pixelIntensity[i];
		}
	}
	
	// normalize intensities so that greatest = 1.0
	for (int i = 0; i < newImg.pixelIntensity.size(); i++) {
		newImg.pixelIntensity[i] /= greatestIntensity;
	}
	
	/* 
	 * sharpening algorithm - find the biggest difference between a pixel and all adjacent
	 *                        set the new value to previous + the biggest difference (close the gap)
	 *
	 */
	for (int i = 0; i < newImg.pixelIntensity.size(); i++) {
		double x = newImg.pixelIntensity[i];
		x += 0.22;
		x = pow(x, 4);
		if (x > 1.0) { x = 1.0; }
		newImg.pixelIntensity[i] = x;
	}
	return newImg;
}

