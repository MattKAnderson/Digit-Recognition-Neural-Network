
/* File to test the code I wrote for the evaluate method */

#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <iostream>

using namespace Eigen;

inline VectorXf cost_derivative(VectorXf &output, VectorXf &expect) {
	return (output - expect);
}

int main() {

	VectorXf vec1(3); 
	vec1 << 3.5, 4.2, 6.7;
	VectorXf vec2(3);
	vec2 << 0.5, 0.2, 0.7;

	VectorXf res = cost_derivative(vec1, vec2);

	std::cout << "Expected vector is: 3, 4, 6...\n";
	std::cout << "Resulting vector is: \n" << res << std::endl;
}


