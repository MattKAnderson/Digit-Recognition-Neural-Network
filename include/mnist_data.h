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
/*
	static bool cmpXCoord(const pixel& p1, const pixel& p2);
	static bool cmpYCoord(const pixel& p2, const pixel& p2);
*/
	/*
     *  2 - dimensional tree class, all planes are split over median integer.
	 *  used for efficiently finding all neighbours around a 2D point.
	 */
/*	class kd_tree {
	public:
		kd_tree(std::vector<pixel>& pixels);
		~kd_tree();

        // Looks for points (x - 1) < p < (x + 1) and similarly for y
		std::vector<pixel> findSetAround(int x, int y);
		node* root;
	private:
		class node { 
		public:
			node(pixel p);
			~node();
			node* leftChild = NULL;
			node* rightChild = NULL;
			pixel pix;
		};

	};
*/
	mnist_data(std::string imgLoc, std::string labelLoc);
    void shuffle();
    int numImages() const;
    int labelAt(int ID) const;
    const image& imgAt(int ID) const;

	static image distortImage(const image& original,
							  double rotationFactor = 0.0,
							  double translationFactor = 0.0,
							  double noiseFactor = 0.0,
							  double elasticFactor = 0.0,
							  uint64_t seed = 0);
private:
    std::vector< std::pair< image, int > > imgLabelPairs;
    int charArrayToInt(char *bytes);
};
#endif
