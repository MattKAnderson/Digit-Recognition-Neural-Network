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

//inlined helper mathematics functions
inline double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
inline double sigmoidPrime(double x) { return sigmoid(x) * (1 - sigmoid(x)); }
inline Eigen::VectorXd hotEncoder(int lbl) {
	Eigen::VectorXd encoded = Eigen::VectorXd::Zero(10);
	encoded[lbl] = 1.0;
	return encoded;
}

class neuralNet {
public:
    neuralNet(std::vector<size_t> sizes);
    int evaluate(const mnist_data& testData);
    void SGD(mnist_data& trainData, 
             size_t epochs, 
             size_t miniBatchSize, 
             double eta,
			 double lambda,
			 mnist_data *testData=NULL);

private:
	typedef std::vector< Eigen::MatrixXd, Eigen::aligned_allocator< Eigen::MatrixXd > > matArray;
	typedef std::vector< Eigen::VectorXd, Eigen::aligned_allocator< Eigen::VectorXd > > vecArray;
    size_t numLayers;
    std::vector<size_t> layerSizes;
    matArray weights;
    vecArray biases;

    //Note: used only for the evaluate method
    Eigen::VectorXd feedforward(const Eigen::VectorXd &inputActivation);

    /*
        @input: activations for first layer
        @expect: expected results corresponding to given input
        @deltaWeights: change in weights from single example
        @deltaBiases: change in biases from single example

        backprop does the back propagation step for a single training example 
        and returns in the last 2 parameters the nudges to the weights and 
        biases corresponding to those steps
    */
    void backprop(const Eigen::MatrixXd &input, 
                  const Eigen::MatrixXd &expect,
                  matArray &deltaWeights,
                  vecArray &deltaBiases);
    
	/* Use examples from train data at indexes from [start, stop) */
    void updateMiniBatch(const mnist_data &trianData, 
                         size_t batchStart, 
                         size_t batchStop, 
                         double eta,
						 double lambda);


    Eigen::MatrixXd quadratic_cost_derivative(const Eigen::MatrixXd &output, 
                                    		  const Eigen::MatrixXd &expect,
											  const Eigen::MatrixXd &inputs);
	
    Eigen::MatrixXd cross_entropy_cost_derivative(const Eigen::MatrixXd &output, 
                                    		  	  const Eigen::MatrixXd &expect);
	void randomizeWeightsBiases();
};

//inlined member functions 
inline Eigen::MatrixXd neuralNet::quadratic_cost_derivative(
												  const Eigen::MatrixXd &output,
												  const Eigen::MatrixXd &expect,
												  const Eigen::MatrixXd &inputs)
{
	return (output - expect).cwiseProduct(inputs.unaryExpr(std::ptr_fun(
											sigmoidPrime)));
}

inline Eigen::MatrixXd neuralNet::cross_entropy_cost_derivative(
													const Eigen::MatrixXd &output,
													const Eigen::MatrixXd &expect)
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
		weights[i] = Eigen::MatrixXd::NullaryExpr(weights[i].rows(),
		                                          weights[i].cols(),
												  [&]() {return normal(rng);});
		biases[i] = Eigen::VectorXd::NullaryExpr(biases[i].rows(), 
												 [&]() {return normal(rng);});
	}
}
//#endif
