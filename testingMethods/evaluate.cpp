/* File to test the code I wrote for the evaluate method */

#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <iostream>

using namespace Eigen;
typedef std::vector<MatrixXf, aligned_allocator<MatrixXf> > matArray;
typedef std::vector<VectorXf, aligned_allocator<VectorXf> > vecArray;

void neuralNet::backprop(Eigen::VectorXf &input, 
                         Eigen::VectorXf &expect,
                         matArray &deltaW,
                         vecArray &deltaB, 
						 int numLayers)
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
	}

	//backward pass
	deltaB[deltaB.size() - 1] = cost_derivative(activations[numLayers - 1], 
	                            expect).cwiseProduct(
								neuronInputs[numLayers - 2].unaryExpr(
							    std::ptr_fun(sigmoidPrime)));
	deltaW[deltaW.size() - 1] = deltaB[deltaB.size() - 1] * 
	                             activations[numLayers - 2].transpose();

	for (int l = 2; l < numLayers - 1; l++) {
		Eigen::VectorXf sp = neuronInputs[numLayers - 1 - l].unaryExpr(std::ptr_fun(sigmoidPrime));
		deltaB[deltaB.size() - l] = weights[weights.size() - l + 1].transpose() 
		                            * deltaB[deltaB.size() - l + 1] * sp;
		deltaW[deltaW.size() - l] = deltaB[deltaB.size() - l] * 
		                            activations[activations.size() - l - 1].transpose();
	}
	//returns deltaW and deltaB as params
}

int main() {
	matArray weights(2);
	weights[0].resize(3,3);
	weights[1].resize(2,3);

	weights[0] << 2, 3, 8,
				  6, 2, 1,
				  4, 6, 5;

	weights[1] << 5, 1, 2,
				  7, 9, 3;

	std::vector<float> testPixels = {0.5, 0.1, 0.9};
	int expected = 1;

	int result = evaluate(testPixels, expected, weights);

	std::cout << "The expected result is: " << expected << std::endl;
	std::cout << "The returned result is: " << result << std::endl;
}
