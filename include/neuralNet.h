//#ifndef NEURAL_NET_H
//#define NEURAL_NET_H
#define EIGEN_USE_MKL_ALL
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <iostream>
#include "mnist_data.h"

//std::vector<Eigen::MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf> > fuck;

//typedef std::vector< Eigen::MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf> > matArray;
//typedef std::vector< Eigen::VectorXf, Eigen::aligned_allocator< Eigen::VectorXf > > vecArray;

/*
To do:
    -constructor         (declared & defined)
    -feedForward         (declared & defined)
    -SGD                 (declared & defined)
    -update_mini_batch   (declared & defined)
    -backProp            (declared & defined)
    -evaluate            (declared & defined)
    -cost_derivative     (declared & defined)
	-randomizeWeightsBiases (declared & defined)
*/
class neuralNet {
public:
    neuralNet(std::vector<size_t> sizes);
    int evaluate(mnist_data& testData);
    void SGD(mnist_data& trainData, 
             size_t epochs, 
             size_t miniBatchSize, 
             float eta,
			 mnist_data *testData=NULL);

private:
	typedef std::vector< Eigen::MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf> > matArray;
	typedef std::vector< Eigen::VectorXf, Eigen::aligned_allocator< Eigen::VectorXf > > vecArray;
    size_t numLayers;
    std::vector<size_t> layerSizes;
    matArray weights;
    vecArray biases;

    //Note: used only for the evaluate method
    Eigen::VectorXf feedforward(Eigen::VectorXf &inputActivation);

    /*
        @input: activations for first layer
        @expect: expected results corresponding to given input
        @deltaWeights: change in weights from single example
        @deltaBiases: change in biases from single example

        backprop does the back propagation step for a single training example 
        and returns in the last 2 parameters the nudges to the weights and 
        biases corresponding to those steps
    */
    void backprop(Eigen::VectorXf &input, 
                  Eigen::VectorXf &expect,
                  matArray &deltaWeights,
                  vecArray &deltaBiases);

    /* Use examples from train data at indexes from [start, stop) */
    void updateMiniBatch(mnist_data &trianData, 
                         size_t batchStart, 
                         size_t batchStop, 
                         float eta);

    Eigen::VectorXf cost_derivative(Eigen::VectorXf &output, 
                                    Eigen::VectorXf &expect);
	
	void randomizeWeightsBiases();
};

//inlined member functions 
inline Eigen::VectorXf neuralNet::cost_derivative(Eigen::VectorXf &output, 
                                                  Eigen::VectorXf &expect) 
{
    return (output - expect);
}
inline void neuralNet::randomizeWeightsBiases() {
	using namespace std::chrono;
	typedef high_resolution_clock clk;
	typedef minutes min;
	uint64_t seed = duration_cast<min>(clk::now().time_since_epoch()).count();
	std::mt19937 rng(seed);
	std::normal_distribution<float> normal(0.0, 1.0);
	for (int i = 0; i < weights.size(); i++) {
		weights[i] = Eigen::MatrixXf::NullaryExpr(weights[i].rows(),
		                                          weights[i].cols(),
												  [&]() {return normal(rng);});
		biases[i] = Eigen::VectorXf::NullaryExpr(biases[i].rows(), 
												 [&]() {return normal(rng);});
	}
}
//inlined helper mathematics functions
inline float sigmoid(float x) { return 1.0 / (1.0 + std::exp(-x)); }
inline float sigmoidPrime(float x) { return sigmoid(x) * (1 - sigmoid(x)); }
inline Eigen::VectorXf hotEncoder(int lbl) {
	Eigen::VectorXf encoded = Eigen::VectorXf::Zero(10);
	encoded[lbl] = 1.0;
	return encoded;
}
//#endif
