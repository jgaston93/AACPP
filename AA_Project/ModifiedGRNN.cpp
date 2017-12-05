#include "ModifiedGRNN.h"
#include <math.h>
#include <iostream>
float MGRNN::EuclideanDistance(float *vec1, float *vec2, float *mask, unsigned int length) {
	unsigned int i;
	float sum = 0;
	for (i = 0; i < length; i++) {
		sum += powf(vec1[i] - vec2[i], 2)*(mask[i]);
		vec1++;
		vec2++;
		mask++;
	}
	return sqrtf(sum);
}

float MGRNN::GaussianKernel(float *vec1, float *vec2, float sigma) {
	float distance = EuclideanDistance(vec1, vec2, Masks + FeatureLength*currentMask, FeatureLength);
	float answer = expf(-((distance*distance) / (2 * sigma*sigma)));
	return answer;
}

void MGRNN::Train(float *data, float *target, unsigned int dataLength, unsigned int featureLength, bool resetMask) {
	if(Data != NULL) delete[] Data;
	if(Target != NULL) delete[] Target;
	if (Masks != NULL) delete[] Masks;
	Data = new float[dataLength*featureLength];
	Target = new float[dataLength];
	unsigned int i;
	for (i = 0; i < dataLength*featureLength; i++) {
		Data[i] = data[i];
	}
	for (i = 0; i < dataLength; i++) {
		Target[i] = target[i];
	}
	FeatureLength = featureLength;
	DataLength = dataLength;
	numMasks = 1;
	currentMask = 0;
	Masks = new float[featureLength];
	if (resetMask) {
		for (i = 0; i < featureLength; i++) {
			Masks[i] = 1;
		}
	}
}

unsigned int MGRNN::Predict(float *instance, unsigned int leaveOut) {
	unsigned int prediction, i;
	float max = 0;
	float strength;
	for (i = 0; i < DataLength; i++) {
		if (i == leaveOut) continue;
		strength = GaussianKernel(instance, Data + FeatureLength*i, Sigma);
		if (strength > max) {
			prediction = Target[i];
			max = strength;
		}
	}
	return prediction;
}

float MGRNN::GRNNPredict(float * instance, unsigned int leaveOut)
{
	unsigned int i;
	float strength, numerator = 0, denominator = 0;
	for (i = 0; i < DataLength; i++) {
		if (i == leaveOut) continue;
		strength = GaussianKernel(instance, Data + FeatureLength*i, Sigma);
		numerator += strength*Target[i];
		denominator += strength;
	}
	denominator += 0.01;

	return numerator / denominator;
}

MGRNN::MGRNN() {
	Data = NULL;
	Target = NULL;
	Masks = NULL;
	DataLength = 0;
	FeatureLength = 0;
	Sigma = 0;
	numMasks = 0;
	currentMask = 0;
}

void MGRNN::SetMask(float *mask, unsigned int numMask) {
	unsigned int i;
	currentMask = 0;
	delete[] Masks;
	Masks = new float[FeatureLength*numMask];
	for (i = 0; i < FeatureLength*numMasks; i++) {
		Masks[i] = mask[i];
	}
}

void MGRNN::SetSigma(float sigma) {
	Sigma = sigma;
}

float MGRNN::LeaveOneOut() {
	unsigned int i, prediction;
	unsigned int correct = 0;
	for (i = 0; i < DataLength; i++) {
		prediction = Predict(Data + FeatureLength*i, i);
		if (prediction == Target[i]) correct++;
	}
	return (float)correct / DataLength;
}

float MGRNN::GRNNLeaveOneOut()
{
	unsigned int i;
	float prediction;
	unsigned int tp = 0, tn = 0, fp = 0, fn = 0, correct;
	for (i = 0; i < DataLength; i++) {
		prediction = GRNNPredict(Data + FeatureLength*i, i);
		if (prediction > 0) {
			if (Target[i] > 0) {
				tp++;
			}
			else {
				fp++;
			}
		}
		else if (prediction < 0) {
			if (Target[i] < 0) {
				tn++;
			}
			else {
				fn++;
			}
		}
	}
	correct = tp + tn;
	return (float)correct / DataLength;
}
