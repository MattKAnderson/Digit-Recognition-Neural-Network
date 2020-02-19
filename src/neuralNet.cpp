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
                    float eta,
					mnist_data *testData)
{
	//iterate over the epochs
	for (int i = 0; i < epochs; i++) {
		trainData.shuffle();
		int batchStart = 0;
		int batchEnd;
		if (miniBatchSize > trainData.numImages()) {
			batchEnd = trainData.numImages();
		} else {
			batchEnd = miniBatchSize;
		}
		while (batchEnd <= trainData.numImages()) {
			updateMiniBatch(trainData, batchStart, batchEnd, eta);
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

int neuralNet::evaluate(mnist_data& testData) 
{
	int success = 0;
	for (int i = 0; i < testData.numImages(); i++) {
		
		//getting input and expected output for test example
		std::vector<float> in = testData.imgAt(i).pixelIntensity;
		Eigen::VectorXf input(layerSizes[0]);
		input = Eigen::Map<Eigen::VectorXf>(in.data(), in.size());	

		Eigen::VectorXf result = this->feedforward(input);

		//get index of max activation in final layer and compare to expected
		Eigen::VectorXf::Index maxI;
		float max = result.maxCoeff(&maxI);
		//std::cout << maxI << "   " << testData.labelAt(i) << std::endl;
		//std::cin.get();
		if(maxI == testData.labelAt(i)) {
			success++;
		}
	}
	return success;
}

/* Definition for backpropagation method */
void neuralNet::backprop(Eigen::VectorXf &input, 
                         Eigen::VectorXf &expect,
                         matArray &deltaW,
                         vecArray &deltaB)
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

	//this isn't even running!!!
	for (int l = 2; l < numLayers; l++) {
		Eigen::VectorXf sp = neuronInputs[numLayers - 1 - l].unaryExpr(std::ptr_fun(sigmoidPrime));
		deltaB[deltaB.size() - l] = (weights[weights.size() - l + 1].transpose() 
		                            * deltaB[deltaB.size() - l + 1]).cwiseProduct(sp);
		deltaW[deltaW.size() - l] = deltaB[deltaB.size() - l] * 
		                            activations[activations.size() - l - 1].transpose();
		//std::cout << "sp: " << sp << "\n\n\n";
		//std::cout << "deltaB: " << deltaB[deltaB.size() - l] << "\n\n\n";
		//std::cout << "deltaW.row(0)[0]: " << deltaW[deltaW.size() - 1].row(0)[0] << "\n";
		//std::cin.get();
	}
	//returns deltaW and deltaB as params
}

void neuralNet::updateMiniBatch(mnist_data &trainData, 
                                size_t batchStart, 
                                size_t batchStop, 
                                float eta)
{
	//initialize the size of the weight and biase adjustments
	matArray deltaW(numLayers - 1);
	vecArray deltaB(numLayers - 1);
	for (int i = 0; i < numLayers - 1; i++) {
		deltaW[i] = Eigen::MatrixXf::Zero(layerSizes[i + 1], layerSizes[i]);
		deltaB[i] = Eigen::VectorXf::Zero(layerSizes[i + 1]);
	}

	//use another set of matArray and vecArray for summing the adjustments
	matArray nablaW(numLayers - 1);
	vecArray nablaB(numLayers - 1);
	for (int i = 0; i < numLayers - 1; i++) {
		nablaW[i] = deltaW[i];
		nablaB[i] = deltaB[i];
	}
	
	//iterate over the batch
	for(int i = batchStart; i < batchStop; i++) {
		
		//getting input and expected output for training example
		std::vector<float> in = trainData.imgAt(i).pixelIntensity;
		Eigen::VectorXf expected = hotEncoder(trainData.labelAt(i));
		Eigen::VectorXf input(layerSizes[0]);
		input = Eigen::Map<Eigen::VectorXf>(in.data(), in.size());	
		//doing backpropagation with training example
		backprop(input, expected, deltaW, deltaB);
		
		//add adjustments to sum of adjustments
		for(int j = 0; j < numLayers - 1; j++) {
			nablaW[j] += deltaW[j];
			nablaB[j] += deltaB[j];
		}
	}

	//average the adjustments
	for (int i = 0; i < numLayers - 1; i++) {
		nablaW[i] /= static_cast<float>(batchStop - batchStart);
		nablaB[i] /= static_cast<float>(batchStop - batchStart);
	}

	//calculate new weights and biases
	for (int i = 0; i < numLayers - 1; i++) {
		weights[i] -= nablaW[i] * eta;
		biases[i] -= nablaB[i] * eta;
	}
}

Eigen::VectorXf neuralNet::feedforward(Eigen::VectorXf &inputActivation) 
{
	Eigen::VectorXf layer;
	layer = inputActivation;
	for(int i = 0; i < numLayers - 1; i++) {
		layer = (weights[i] * layer).unaryExpr(std::ptr_fun(sigmoid));
	}
	return layer;
}
