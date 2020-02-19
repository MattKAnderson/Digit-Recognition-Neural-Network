#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <random>
#include <chrono>
#include <iostream>

int main() {
	using namespace std::chrono;
	typedef high_resolution_clock clk;
	typedef minutes min;

	Eigen::MatrixXf weights(3,3);
	Eigen::VectorXf biases(3);

	uint64_t seed = duration_cast<min>(clk::now().time_since_epoch()).count();
	std::mt19937 rng(seed);
	std::normal_distribution<float> normal(0.0, 1.0);

	weights = Eigen::MatrixXf::NullaryExpr(weights.rows(),
	                                          weights.cols(),
											  [&]() {return normal(rng);});
	biases = Eigen::VectorXf::NullaryExpr(biases.rows(), 
											 [&]() {return normal(rng);});

	std::cout << "The elements of weights are: \n" << weights << std::endl;
	std::cout << "\nThe elements of biases are: \n" << biases << std::endl;

}
