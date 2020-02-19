#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <iostream>

inline Eigen::VectorXf hotEncoder(int lbl) {
	Eigen::VectorXf encoded = Eigen::VectorXf::Zero(10);
	encoded[lbl] = 1.0;
	return encoded;
}

int main() {
	int five = 5;
	Eigen::VectorXf encoded5 = hotEncoder(five);

	std::cout << "Five encoded is: \n" << encoded5 << std::endl;
	
	Eigen::VectorXf::Index maxI;
	float max = encoded5.maxCoeff(&maxI);

	std::cout << "\n\nIndex is: " << maxI << std::endl;
}
