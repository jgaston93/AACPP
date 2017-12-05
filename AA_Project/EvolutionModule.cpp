#include "EvolutionModule.h"
#include <stdlib.h>
#include <algorithm>
#include <random>

float RandomFloatRange(float a, float b) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(a, b);
	return (float)dis(gen);
}

void InitializePopulation(float * fitnessVector, float * masks, float * sigmas, unsigned int populationSize, unsigned int featureLength)
{
	unsigned int i;
	for (i = 0; i < populationSize; i++) {
		fitnessVector[i] = 0;
		sigmas[i] = RandomFloatRange(0, 1);
	}
	for (i = 0; i < populationSize*featureLength; i++) {
		masks[i] = RandomFloatRange(0, 1);
	}
}

void EvaluatePopulation(MGRNN clf, float * fitnessVector, float * masks, float * sigmas, unsigned int populationSize, unsigned int featureLength, bool diagnostic)
{
	unsigned int i,j;
	for (i = 0; i < populationSize; i++) {
		clf.SetMask(masks + featureLength*i, 1);
		clf.SetSigma(sigmas[i]);
		fitnessVector[i] = clf.GRNNLeaveOneOut();
		if (diagnostic) {
			for (j = 0; j < featureLength; j++) {
				printf("%f ", (masks + featureLength*i)[j]);
			}
			printf("\nSigma: %f\nFitness: %f\n\n", sigmas[i], fitnessVector[i]);
		}
	}
}

unsigned int GetLeastFitParent(float * fitnessVector, unsigned int length)
{
	unsigned int i, minIndex;
	float minFitness = 1;
	for (i = 0; i < length; i++) {
		if (fitnessVector[i] < minFitness) {
			minIndex = i;
			minFitness = fitnessVector[i];
		}
	}
	return minIndex;
}

void ReplaceParent(float * fitnessVector, float * masks, float * sigmas, float childFitness, float * childMask, float childSigma, unsigned int populationSize, unsigned int featureLength)
{
	unsigned int i, parent = GetLeastFitParent(fitnessVector, populationSize);
	fitnessVector[parent] = childFitness;
	sigmas[parent] = childSigma;
	for (i = 0; i < featureLength; i++) {
		masks[featureLength*parent + i] = childMask[i];
	}
}

void BLXCrossover(float * vec1, float * vec2, float * vec3, unsigned int length, float extend)
{
	unsigned int i;
	float a, b;
	for (i = 0; i < length; i++) {
		if (vec1[i] < vec2[i]) {
			a = vec1[i];
			b = vec2[i];
		}
		else {
			a = vec2[i];
			b = vec1[i];
		}
		a -= extend;
		b += extend;
		if (a < 0) a = 0;
		if (b > 1) b = 1;
		vec3[i] = RandomFloatRange(a, b);
	}
}

unsigned int TournamentSelection(float *fitnessVector, unsigned int length, unsigned int groupSize) {
	unsigned int i, maxIndex = 0, currentIndex;
	float maxFitness = 0;
	std::vector<unsigned int> indices(length);
	for (i = 0; i < length; i++) {
		indices[i] = i;
	}
	std::random_device rd;
	std::mt19937 gen(rd());
	std::shuffle(indices.begin(), indices.end(), gen);
	for (i = 0; i < groupSize; i++) {
		currentIndex = indices[i];
		if (fitnessVector[currentIndex] > maxFitness) {
			maxFitness = fitnessVector[currentIndex];
			maxIndex = currentIndex;
		}
	}
	return maxIndex;
}

void GenerationAnalytics(float *fitnessVector, unsigned int length, float *max, float *min, float *avg) {
	unsigned int i;
	float maxFitness = 0, minFitness = 1, avgFitness = 0;
	for (i = 0; i < length; i++) {
		if (fitnessVector[i] < minFitness) minFitness = fitnessVector[i];
		if (fitnessVector[i] > maxFitness) maxFitness = fitnessVector[i];
		avgFitness += fitnessVector[i];
	}
	*max = maxFitness;
	*min = minFitness;
	*avg = avgFitness / (float)length;
}


void Evolve(MGRNN clf, unsigned int generations, unsigned int populationSize, unsigned int featureLength)
{
	float *FitnessVector = new float[populationSize];
	float *Masks = new float[populationSize*featureLength];
	float *Sigmas = new float[populationSize];
	float *ChildMask = new float[featureLength];
	float ChildSigma, ChildFitness;
	float Max, Min, Avg;

	unsigned int parent1, parent2, generation;
	InitializePopulation(FitnessVector, Masks, Sigmas, populationSize, featureLength);
	EvaluatePopulation(clf, FitnessVector, Masks, Sigmas, populationSize, featureLength, true);
	for (generation = 0; generation < generations; generation++) {
		GenerationAnalytics(FitnessVector, populationSize, &Max, &Min, &Avg);
		printf("Generation: %d\n\tMax: %f\n\tAvg: %f\n\tMin: %f\n\n", generation, Max, Avg, Min);
		parent1 = TournamentSelection(FitnessVector, populationSize, 2);
		parent2 = TournamentSelection(FitnessVector, populationSize, 2);
		BLXCrossover(Masks + featureLength*parent1, Masks + featureLength*parent2, ChildMask, featureLength, 0);
		BLXCrossover(Sigmas + parent1, Sigmas + parent2, &ChildSigma, 1, 0);
		EvaluatePopulation(clf, &ChildFitness, ChildMask, &ChildSigma, 1, featureLength, false);
		ReplaceParent(FitnessVector, Masks, Sigmas, ChildFitness, ChildMask, ChildSigma, populationSize, featureLength);
	}

	FILE *fp;
	unsigned int i, j;
	fp = fopen("data/featuremasks.csv", "w");
	fprintf(fp, "Accuracy,Sigma,FeatureMask\n");
	for (i = 0; i < populationSize; i++) {
		fprintf(fp, "%f,%f,", FitnessVector[i], Sigmas[i]);
		for (j = 0; j < featureLength; j++) {
			fprintf(fp, "%f ", Masks[featureLength*i + j]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
}