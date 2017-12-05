#pragma once
class MGRNN {
private:
	float *Data;
	float *Target;
	float *Masks;
	unsigned int currentMask;
	unsigned int FeatureLength;
	unsigned int DataLength;
	unsigned int numMasks;
	float Sigma;
	float GaussianKernel(float *vec1, float *vec2, float sigma);
	float EuclideanDistance(float *vec1, float *vec2, float *mask, unsigned int length);
public:
	MGRNN();
	void SetMask(float *mask, unsigned int numMask);
	void SetSigma(float sigma);
	void Train(float *data, float *target, unsigned int dataLength, unsigned int featureLength, bool resetMask);
	unsigned int Predict(float *instance, unsigned int leaveOut);
	float GRNNPredict(float *instance, unsigned int leaveOut);
	float LeaveOneOut();
	float GRNNLeaveOneOut();
};