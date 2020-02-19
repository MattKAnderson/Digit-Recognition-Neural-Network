
/* File to test the code I wrote for the evaluate method */

#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <iostream>
#include <cmath>

using namespace Eigen;
typedef std::vector<MatrixXf, aligned_allocator<MatrixXf> > matArray;
typedef std::vector<VectorXf, aligned_allocator<VectorXf> > vecArray;
inline VectorXf cost_derivative(VectorXf &output, VectorXf &expect) {
	return (output - expect);
}

inline float sigmoid(float x) { return 1.0 / (1.0 + std::exp(-x)); }
inline float sigmoidPrime(float x) { return sigmoid(x) * (1 - sigmoid(x)); }

void backprop(Eigen::VectorXf &input, 
                         Eigen::VectorXf &expect,
                         matArray &deltaW,
                         vecArray &deltaB,
						 int numLayers,
						 matArray &weights)
{
	//zero the change in weights and bias changes
	for (int i = 0; i < deltaW.size(); i++) {
		deltaW[i] = Eigen::MatrixXf::Zero(deltaW[i].rows(), deltaW[i].cols());		
		deltaB[i] = Eigen::VectorXf::Zero(deltaB[i].rows(), deltaB[i].cols());
	}

	// feed forward and store all activations and neuron inputs layer by layer
	vecArray activations(numLayers);
	vecArray neuronInputs(numLayers - 1);
	activations[0] = input;
	for (int i = 1; i < numLayers; i++) {
		neuronInputs[i-1] = weights[i-1] * activations[i-1];
		activations[i] = neuronInputs[i - 1].unaryExpr(std::ptr_fun(sigmoid)); 
		std::cout << activations[i] << std::endl << std::endl;
	}

	//backward pass
	deltaB[deltaB.size() - 1] = cost_derivative(activations[numLayers - 1], 
	                            expect).cwiseProduct(
								neuronInputs[numLayers - 2].unaryExpr(
							    std::ptr_fun(sigmoidPrime)));
	deltaW[deltaW.size() - 1] = deltaB[deltaB.size() - 1] * 
	                             activations[numLayers - 2].transpose();

	std::cout << "\n\nDeltaB[-1] =\n" << deltaB[deltaB.size() - 1] << std::endl;
	std::cout << "\nDeltaW[-1] =\n" << deltaW[deltaW.size() - 1] << std::endl;
	for (int l = 2; l < numLayers; l++) {
		Eigen::VectorXf sp = neuronInputs[neuronInputs.size() - l].unaryExpr(std::ptr_fun(sigmoidPrime));
		VectorXf temp = weights[weights.size() - l + 1].transpose() * deltaB[deltaB.size() - l + 1];
		std::cout << "\n\nTemp: \n" << temp << std::endl;
		deltaB[deltaB.size() - l] = (weights[weights.size() - l + 1].transpose() 
		                            * deltaB[deltaB.size() - l + 1]).cwiseProduct(sp);
		deltaW[deltaW.size() - l] = deltaB[deltaB.size() - l] * 
		                            activations[activations.size() - l - 1].transpose();
		std::cout << "\n\nDeltaB[-2] =\n" << deltaB[deltaB.size() - l] << std::endl;
		std::cout << "\nDeltaW[-2] =\n" << deltaW[deltaW.size() - l] << std::endl;
	}
	//returns deltaW and deltaB as params
}

int main() {
	matArray weights(2);
	matArray dw(2);
	vecArray db(2);
	weights[0].resize(3,3);
	weights[1].resize(3,3);
	
	weights[0] << 0.2, 0.3, 0.8,
				  0.6, 0.2, 1.1,
				  0.4, 0.6, 1.5;

	weights[1] << 0.5, -1.0, 1.2,
				  -0.7, 0.9, 0.3,
				  -1.3, -1.4, 1.0;

	dw[0] = weights[0];
	dw[1] = weights[1];
	db[0].resize(3);
	db[1].resize(3);
	VectorXf in(3);
	in << 0.8, 0.2, 0.3;

	VectorXf expected(3);
	expected << 0, 1, 0;


	backprop(in, expected, dw, db, 3, weights);

}
