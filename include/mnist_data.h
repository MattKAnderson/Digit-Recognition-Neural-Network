#ifndef MNIST_DATA_H
#define MNIST_DATA_H

#include <vector>
#include <string>
#include <fstream>
#include <random>
#include <chrono>
#include <algorithm>
#include <utility>

struct image { std::vector< float > pixelIntensity; }

class mnist_data {
public:
    mnist_data(std::string imgLoc, std::string labelLoc);
    void shuffle();
    int numImages();
    int labelAt(int ID);
    image& imgAt(int ID);

private:
    std::vector< std::pair< image, int > > imgLabelPairs;
    int charArrayToInt(char *bytes);
}
#endif
