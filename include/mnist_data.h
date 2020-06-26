#ifndef MNIST_DATA_H
#define MNIST_DATA_H

#include <vector>
#include <string>
#include <fstream>
#include <random>
#include <chrono>
#include <algorithm>
#include <utility>


class mnist_data {
public:

	// struct to represent a b/w image as a vector of its pixel's intensities
	struct image { 
		std::vector< double > pixelIntensity; 
	};

	// struct to represent a pixel (i.e position and intensity)
	struct pixel {
		double x;
		double y;
		double intensity;
	};
	
	mnist_data(std::string imgLoc, std::string labelLoc);
    void shuffle();
    int numImages() const;
    int labelAt(int ID) const;
    const image& imgAt(int ID) const;
	void expandWithDistortions(int factor, 
							   double rotationFactor, 
							   uint64_t seed);

	static image distortImage(const image& original,
							  double rotationFactor,
							  std::mt19937& gen);

private:
    std::vector< std::pair< image, int > > imgLabelPairs;
    int charArrayToInt(char *bytes);
};
#endif
