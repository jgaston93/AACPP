#include "ModifiedGRNN.h"
#include "DataLoader.h"
#include "EvolutionModule.h"
#include <iostream>
using namespace std;

int main() {
	MGRNN clf = MGRNN();
	LoadDataset(clf, "data/our_dataset.txt");
	//clf.SetSigma(0.11853);
	//printf("Accuracy: %f", clf.GRNNLeaveOneOut());
	Evolve(clf, 180, 20, 95);

}