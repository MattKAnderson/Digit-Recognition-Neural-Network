#include "neuralNet.h"
neuralNet::neuralNet(std::vector<size_t> sizes) 
{
	layerSizes = sizes;
	numLayers = sizes.size();
	weights.resize(numLayers - 1);
	biases.resize(numLayers - 1);
	for (int i = 0; i < numLayers - 1; i++) {
		weights[i].resize(sizes[i + 1], sizes[i]);
		biases[i].resize(sizes[i + 1]);
	}
	randomizeWeightsBiases();
}

void neuralNet::SGD(mnist_data& trainData, 
                    size_t epochs, 
                    size_t miniBatchSize, 
                    double eta,
					double lambda,
					mnist_data *testData)
{
	//iterate over the epochs
	for (int i = 1; i <= epochs; i++) {
		if (i % 100 == 0) { eta *= 0.3; }
		trainData.shuffle();
		int batchStart = 0;
		int batchEnd;
		if (miniBatchSize > trainData.numImages()) {
			batchEnd = trainData.numImages();
		} else {
			batchEnd = miniBatchSize;
		}
		while (batchEnd <= trainData.numImages()) {
			updateMiniBatch(trainData, batchStart, batchEnd, eta, lambda);
			batchStart += miniBatchSize;
			batchEnd += miniBatchSize;
		}
		if (testData) {
			std::cout << "Epoch " << i << ":  " << evaluate(*testData) << " / " 
			          << testData->numImages() << std::endl;
		} else {
			std::cout << "Epoch " << i << " complete..." << std::endl;
		}
	}
}

int neuralNet::evaluate(const mnist_data& testData) 
{
	int success = 0;
	for (int i = 0; i < testData.numImages(); i++) {
		
		//getting input and expected output for test example
		std::vector<double> in = testData.imgAt(i).pixelIntensity;
		Eigen::VectorXd input(layerSizes[0]);
		input = Eigen::Map<Eigen::VectorXd>(in.data(), in.size());	

		Eigen::VectorXd result = this->feedforward(input);

		//get index of max activation in final layer and compare to expected
		Eigen::VectorXd::Index maxI;
		double max = result.maxCoeff(&maxI);
		if(maxI == testData.labelAt(i)) {
			success++;
		}
	}
	return success;
}

void neuralNet::backprop(const Eigen::MatrixXd &input,
						 const Eigen::MatrixXd &expect,
						 matArray &nablaW,
						 vecArray &nablaB)
{
	//feed forward
	Eigen::MatrixXd bias(input.rows(),input.cols());
	Eigen::MatrixXd delta;
	matArray activations(numLayers);
	matArray neuronInputs(numLayers - 1);
	activations[0] = input;

	for (int i = 1; i < numLayers - 1; i++) {
		bias.colwise() = biases[i-1];
		neuronInputs[i-1] = (weights[i-1] * activations[i-1]) + bias;
		activations[i] = neuronInputs[i-1].unaryExpr(std::ptr_fun(sigmoid));
	}

	// softmax final layer
	bias.colwise() = biases[numLayers - 2];
	neuronInputs[numLayers - 2] = (weights[numLayers - 2] * activations[numLayers - 2]) + bias;
	activations[numLayers - 1] = neuronInputs[numLayers - 2].array().exp();
	for (int i = 0; i < activations.back().cols(); i++) {
		double sum = activations.back().col(i).sum();
		//std::cout << sum << std::endl;
		//std::cin.get();
		activations.back().col(i) /= sum;
	}

	//backwards pass
	//delta = quadratic_cost_derivative(activations[numLayers - 1], expect, neuronInputs[numLayers - 2]);
	delta = cross_entropy_cost_derivative(activations[numLayers - 1], expect);
	nablaB[nablaB.size() - 1] = delta.rowwise().sum();
	nablaW[nablaW.size() - 1] = delta * activations[numLayers - 2].transpose();
	for (int l = 2; l < numLayers; l++) {
		Eigen::MatrixXd sp = neuronInputs[numLayers - 1 - l].unaryExpr(
		                      std::ptr_fun(sigmoidPrime));
		delta = (weights[weights.size() - l + 1].transpose() * 
		         delta).cwiseProduct(sp);
		nablaB[nablaB.size() - l] = delta.rowwise().sum();
		nablaW[nablaW.size() - l] = delta * activations[
		                             activations.size() - l - 1].transpose();
	}
}

void neuralNet::updateMiniBatch(const mnist_data &trainData, 
                                size_t batchStart, 
                                size_t batchStop, 
                                double eta,
								double lambda)
{
	size_t size = batchStop - batchStart;

	//initialize the size of the weight and biase adjustments
	matArray nablaW(numLayers - 1);
	vecArray nablaB(numLayers - 1);

	//declaring the input and expected matrices
	Eigen::MatrixXd input(layerSizes[0], size);
	Eigen::MatrixXd expected(layerSizes[numLayers- 1], size);

	//generate the input and expected matrices of the training examples
	int pos = 0;
	for(int i = batchStart; i < batchStop; i++) {
		std::vector<double> in = trainData.imgAt(i).pixelIntensity;
		input.col(pos) = Eigen::Map<Eigen::VectorXd>(in.data(), in.size());	
		expected.col(pos) = hotEncoder(trainData.labelAt(i));
		pos++;
	}
	backprop(input, expected, nablaW, nablaB);
	
	//calculate new weights and biases using L2 regularization
	for (int i = 0; i < numLayers - 1; i++) {
		weights[i] = (1 - eta * lambda / static_cast<double>(trainData.numImages())) * weights[i]
		             - (nablaW[i] / static_cast<double>(size)) * eta;
		biases[i] -= (nablaB[i] / static_cast<double>(size)) * eta;
	}
}

Eigen::VectorXd neuralNet::feedforward(const Eigen::VectorXd &inputActivation) 
{
	Eigen::VectorXd layer;
	layer = inputActivation;
	int i;
	for(i = 0; i < numLayers - 2; i++) {
		layer = ((weights[i] * layer) + biases[i]).unaryExpr(std::ptr_fun(sigmoid));
	}
	layer = ((weights[i] * layer) + biases[i]).array().exp();
	layer /= layer.maxCoeff();
	double sum = layer.sum();
	layer /= sum;
	return layer;
}
