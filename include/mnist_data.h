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

	/*!
	 *  Returns the number of images in the dataset
	 */
	int numImages() const;

	/*!
	 *  Returns the label at the given position in the imgLabelPair vector.
	 *  @Param ID The index of the label to get
	 *  @Return The label of the image - label pair at the given index
	 */
    int labelAt(int ID) const;

	/*!
	 *  Returns the image at the given position in the imgLabelPair vector.
	 *  @Param ID The index of the image to get
	 *  @Return The image of the image - label pair at the given index
	 */
    const image& imgAt(int ID) const;

	/*!
	 *	Expands this data set by a given factor (n) by distorting each image
	 *  (n-1) times. Calls distortImage to produce additional images.
	 *  @Param factor The factor to increase the data set by (i.e. produce
	 *                factor - 1 new images through distortion of originals)
	 *  @param rotationFactor Standard deviation of distribution random 
	 *                        rotation angles are pulled from when creating
	 *                        distorted images
	 *  @param seed Unsigned 64 bit integer used to seed the random number 
	 *              generator
	 */
	void expandWithDistortions(int factor, 
							   double rotationFactor, 
							   uint64_t seed);

	/*!
     *  Produces a distorted image of an original. Rotates image by a random
	 *  angle according to a normal distribution with standard deviation given
	 *  by the rotation factor and returns it as a new image.
	 *  @Param original The original image object
	 *  @Param rotationFactor The standard deviation for the random angle that
	 *                        that is used to rotate the image
	 *  @Param gen A seeded Mersenne twister (mt19937) prng object
	 *  @return A new image with a random angle applied
	 */
	static image distortImage(const image& original,
							  double rotationFactor,
							  std::mt19937& gen);

private:
    std::vector< std::pair< image, int > > imgLabelPairs;
    int charArrayToInt(char *bytes);
};
#endif
