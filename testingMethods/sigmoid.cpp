#include <iostream>
#include <cmath>
inline float sigmoid(float x) { return 1.0 / (1.0 + std::exp(-x)); }

int main() {

	float input = -3.4143357;

	float result = sigmoid(input);
	float expected = 0.03186;

	std::cout << "The expected result is: " << expected << "\nthe calculated "
	          << "result is: " << result << std::endl;


}
