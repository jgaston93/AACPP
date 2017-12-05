#include "DataLoader.h"
#include <fstream>
#include <stdlib.h>
using namespace std;

void LoadDataset(MGRNN &clf, const char filename[]) {
	ifstream dataFile;
	dataFile.open(filename);
	char buffer[64];
	char c;
	unsigned int bufferIndex = 0;
	unsigned int featureLength, dataLength;

	// Read the file paramters
	while (dataFile.get(c)) {
		if (c != ' ' && c != '\n') {
			buffer[bufferIndex++] = c;
		}
		else {
			buffer[bufferIndex] = '\0';
			if (c == '\n') {
				featureLength = atoi(buffer);
				printf("Feature Length: %d\n", featureLength);
				break;
			}
			else {
				dataLength = atoi(buffer);
				printf("Data Length: %d\n", dataLength);
				bufferIndex = 0;
			}
		}
	}
	float *data = new float[dataLength*featureLength];
	float *target = new float[dataLength];
	bufferIndex = 0;
	unsigned int dataIndex = 0, targetIndex = 0, instanceIndex = 0;
	unsigned int i, featureCounter;
	float sum, magnitude;
	while (dataFile.get(c)) {
		// Skip the instance index
		while (dataFile.get() != ' ');
		// Read the label
		while (dataFile.get(c) && c != ' ') {
			buffer[bufferIndex++] = c;
		}
		buffer[bufferIndex] = '\0';
		target[targetIndex++] = atof(buffer);
		//printf("Target: %f\n", target[targetIndex - 1]);
		bufferIndex = 0;
		// Read the features
		sum = 0;
		featureCounter = 0;
		while (dataFile.get(c) && c != '\n') {
			if (c != ' ') {
				buffer[bufferIndex++] = c;
			}
			else {
				buffer[bufferIndex] = '\0';
				data[dataIndex] = atof(buffer);
				sum += data[dataIndex] * data[dataIndex];
				dataIndex++;
				bufferIndex = 0;
				featureCounter++;
			}
		}
		if (featureCounter != 95) printf("Something Went wrong\n");
		/*buffer[bufferIndex] = '\0';
		data[dataIndex] = atof(buffer);
		sum += data[dataIndex] * data[dataIndex];
		dataIndex++;*/
		bufferIndex = 0;
		magnitude = sqrtf(sum);
		if (magnitude != 0) {
			for (i = 0; i < featureLength; i++) {
				data[featureLength*instanceIndex + i] /= magnitude;
			}
		}
		instanceIndex++;
	}
	clf.Train(data, target, dataLength, featureLength, true);
	delete[] data;
	delete[] target;
}