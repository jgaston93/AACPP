#pragma once
#include "ModifiedGRNN.h"

void Evolve(MGRNN clf, unsigned int generations, unsigned int populationSize, unsigned int featureLength);
void BLXCrossover(float *vec1, float *vec2, float *vec3, unsigned int length, float extend);
unsigned int TournamentSelection(float *fitnessVector, unsigned int length, unsigned int groupSize);
float RandomFloatRange(float a, float b);
void InitializePopulation(float *fitnessVector, float *masks, float *sigmas, unsigned int populationSize, unsigned int featureLength);
void EvaluatePopulation(MGRNN clf, float *fitnessVector, float *masks, float *sigmas, unsigned int populationSize, unsigned int featureLength, bool diagnostic);
unsigned int GetLeastFitParent(float *fitnessVector, unsigned int length);
void ReplaceParent(float *fitnessVector, float *masks, float *sigmas, float childFitness, float *childMask, float childSigma, unsigned int populationSize, unsigned int featureLength);