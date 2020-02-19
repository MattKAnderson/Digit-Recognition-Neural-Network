/* File to test the code I wrote for the feed-forward method */

#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <iostream>

using namespace Eigen;
typedef std::vector<MatrixXf, aligned_allocator<MatrixXf> > matArray;

VectorXf feedforward(VectorXf &in, matArray &weights) {
	VectorXf out;
	out = in;
	for(int i = 0; i < weights.size(); i++) {
		out = weights[i] * out;
	}
	return out;
}

int main() {
	matArray weights(2);
	weights[0].resize(3,3);
	weights[1].resize(2,3);
	VectorXf in(3);
	VectorXf expect(2);

	weights[0] << 2, 3, 8,
				  6, 2, 1,
				  4, 6, 5;

	weights[1] << 5, 1, 2,
				  7, 9, 3;

	in << 1, 2, 3;

	expect << 235, 434;

	VectorXf result = feedforward(in, weights);

	std::cout << "The expected vector is:\n" << expect << std::endl;
	std::cout << "The resulting vector is:\n" << result << std::endl;



}




