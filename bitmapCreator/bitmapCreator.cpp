#include <algorithm>
#include "bitmapCreator.h"

void bitmapCreator::writeBitmap(std::string filename, 
								std::vector<int>& imgData, 
								int rowSize)
{
	int height = imgData.size() / rowSize;
	int bits = rowSize * 24;
	int rowPadding = (bits % 32) / 8;
	std::ofstream file(filename, std::ios::out | std::ios::binary);
	
	writeBMPHeader(file, rowSize, height);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < rowSize; j++) {	
			int val = imgData[rowSize * i + j];
			char pixelRGB[3] = { val, val, val };
			file.write(pixelRGB, 3);
		}
		for (int j = 0; j < rowPadding; j++) { 
			char paddingByte[1] = { 0 };
			file.write(paddingByte, 1);
		}
	}
	file.close();
}

void bitmapCreator::writeBMPHeader(std::ofstream& file, int width, int height) {
	
	// calculating number of bytes for the image data
	int bits = width * 24;
	int rowPadding = (bits % 32) / 8;
	int bytesPerRow = (bits + rowPadding) / 8;
	int totalPixelBytes = bytesPerRow * height;
	int totalSize = totalPixelBytes + 54;
	std::vector<int> sizeInHex = decimalToHexadecimal(totalSize);

	// error if size is too large
	if (sizeInHex.size() > 8) { 
		// THROW ERROR HERE
	}

	/*
	 *  creating byte array to write to file
	 */
	char* header = new char[54];
	header[0] = 0x42;
	header[1] = 0x4D;
	int pos = 2;
	
	// writing file size in bytes in little endian form
	for (int i = sizeInHex.size() - 1; i >= 0; i -= 2) {
		int value = sizeInHex[i];
		if (i > 0) {
			value += sizeInHex[i-1] * 16;
		}
		header[pos] = static_cast<char>(value);
		pos++;
	}
	while (pos != 6) { 
		header[pos] = 0;
		pos++;
	}
		
	//next 4 bytes unused
	while (pos < 10) {
		header[pos] = 0;
		pos++;
	}

	// offset to bitmap data (4 bytes, little endian)
	header[10] = 0x36;
	for (int i = 11; i < 14; i++) { header[i] = 0; }

	// DIB header
	header[14] = 0x28;
	for (int i = 15; i < 18; i++) { header[i] = 0; }
	
	// writing width of image in hexadecimal, little endian form
	std::vector<int> widthInHex = decimalToHexadecimal(width);
	pos = 18;
	for (int i = widthInHex.size() - 1; i >= 0; i -= 2) {
		int value = widthInHex[i];
		if (i > 0) {
			value += widthInHex[i-1] * 16;
		}
		header[pos] = static_cast<char>(value);
		pos++;
	}
	while (pos != 22) { 
		header[pos] = 0;
		pos++;
	}

	// writing height of image in hexadecimal, little endian form
	std::vector<int> heightInHex = decimalToHexadecimal(height);
	complementBytes(heightInHex);
	for (int i = heightInHex.size() - 1; i >= 0; i -= 2) {
		int value = heightInHex[i];
		if (i > 0) {
			value += heightInHex[i-1] * 16;
		}
		header[pos] = static_cast<char>(value);
		pos++;
	}
	while (pos != 26) { 
		header[pos] = 0;
		pos++;
	}
	header[26] = 1;
	header[27] = 0;
	header[28] = 0x18;
	for (pos = 29; pos < 34; pos++) { header[pos] = 0; }
	
	// writing size of raw image data in bytes, little endian form
	std::vector<int> imgDataSizeInHex = decimalToHexadecimal(totalPixelBytes);
	for (int i = imgDataSizeInHex.size() - 1; i >= 0; i -= 2) {
		int value = imgDataSizeInHex[i];
		if (i > 0) {
			value += imgDataSizeInHex[i-1] * 16;
		}
		header[pos] = static_cast<char>(value);
		pos++;
	}
	while (pos != 38) { 
		header[pos] = 0;
		pos++;
	}

	header[38] = 0x13;
	header[39] = 0x0B;
	header[40] = 0;
	header[41] = 0;
	header[42] = 0x13;
	header[43] = 0x0B;
	for (pos = 44; pos < 54; pos++) { header[pos] = 0; }

	// write to the file
	file.write(header, 54);
	delete header;
}

void bitmapCreator::complementBytes(std::vector<int>& semiBytes) {
	if (semiBytes.size() == 0) { 
		for (int i = 0; i < 8; i++) { semiBytes.push_back(15); } 
	}
	else {
		std::reverse(semiBytes.begin(), semiBytes.end());
		semiBytes[0] = 16 - semiBytes[0];
		int ind = 1;
		while (ind < semiBytes.size()) {
			semiBytes[ind] = 15 - semiBytes[ind];
			ind++;
		}
		while (ind < 8) { semiBytes.push_back(15); ind++; }
		std::reverse(semiBytes.begin(), semiBytes.end());
	}
}

std::vector<int> bitmapCreator::decimalToHexadecimal(int num) {
	std::vector<int> digits;
	while (num != 0) {
		digits.push_back(num % 16);
		num /= 16;
	}
	std::reverse(digits.begin(), digits.end());
	return digits;
}
