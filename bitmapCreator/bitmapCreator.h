/* class to write a greyscale image to a bitmap image */

#ifndef BITMAPCREATOR_H
#define BITMAPCREATOR_H

#include <fstream>
#include <vector>
#include <string>

class bitmapCreator {
public:
	bitmapCreator() {};
	void writeBitmap(std::string filename, 
					 std::vector<int>& imgData, 
					 int rowSize);

private:
	void writeBMPHeader(std::ofstream& file, int width, int height);
	void complementBytes(std::vector<int>& bytes);
	std::vector<int> decimalToHexadecimal(int num);
};

#endif
