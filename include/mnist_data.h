#ifndef MNIST_DATA_H
#define MNIST_DATA_H

#include <vector>
#include <string>
#include <fstream>
#include <random>
#include <chrono>
#include <algorithm>
#include <utility>

struct image { std::vector< double > pixelIntensity; };

class mnist_data {
public:
    mnist_data(std::string imgLoc, std::string labelLoc);
    void shuffle();
    int numImages() const;
    int labelAt(int ID) const;
    const image& imgAt(int ID) const;

private:
    std::vector< std::pair< image, int > > imgLabelPairs;
    int charArrayToInt(char *bytes);
};
#endif
