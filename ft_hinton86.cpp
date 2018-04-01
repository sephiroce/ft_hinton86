/**
 student id: 2016-30729 / name: Kyungmin Lee
 update: 20161005 
  - Weight update rules were improved(?)
  - print out the best distributed representations
  - minor bug fix: valid set index
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>     /* srand, rand */

#define N_PERSON 24		//size of one-hot encoding of persons
#define N_RELATION 12	//size of one-hot encoding of relations
#define N_DEP 6			//size of distributed encoding of persons
#define N_DER 6			//size of distributed encoding of relations
#define N_TRAIN 100		//size of training set
#define N_VALID   4		//size of validation set
#define N_CENTRAL 12		//size of central layer
#define MAX_STR 20

#define NEGLIGIBLE_ERROR 0		//stop condition 
#define MAX_ITER 		 1500	//stop condition, accoring to Hinton's paper 573 for reducing error to be negligible or 1500 for learing distributed encodings, 

//#define DEBUG

const char names[N_PERSON][MAX_STR]
= { "Christopher", "Penelope", "Andrew", "Christine", "Margaret", "Arthur",
"Colin", "Roberto", "Maria", "Gina", "Emilio", "Alfonso",
"Lucia", "Marco",	"Sophia", "Victoria", "James", "Charlotte",
"Pierro", "Francesca", "Angela", "Tomaso", "Jennifer", "Charles" };

const char relation[N_RELATION][MAX_STR]
= { "son", "daughter", "nephew", "niece",
"father", "mother", "uncle", "aunt",
"brother", "sister", "husband", "wife" };

float LR = 0.35f;		//Learning Rate
float ALPHA = 0.33f;		//Exponential decay factor 
#define SLOPE_SIGMOID 1//slope of sigmoid functions

//5 layers: all layers are sigmoid ?
float outputPersonVec[N_TRAIN][N_PERSON];
float outputPersonDE[N_TRAIN][N_DEP];
float centralLayer[N_TRAIN][N_CENTRAL];
float inputDE[N_TRAIN][N_DEP + N_DER];	//The second layer has two groups
float min_VE = 100.0f;

//Training Set
int trainSet[N_TRAIN][3];	//0: Person1, 1: Relation, 2: Person2

//Validation Set
int validSet[N_VALID][3];	//0: Person1, 1: Relation, 2: Person2

//5 weight matrix
float w_pv2de[N_PERSON][N_DEP];				//weights Person Vector				  -> Person Distributed Encoding(Encoding)
float min_w_pv2de[N_PERSON][N_DEP];
float w_rv2de[N_RELATION][N_DER];			//weights Relation Vector             -> Relation Distributed Encoding(Encoding)
float min_w_rv2de[N_RELATION][N_DER];
float w_de2cl[N_DEP + N_DER][N_CENTRAL];	//weights Distributed Encoding		  -> Central layer
float w_cl2pd[N_CENTRAL][N_DEP];			//weights Central layer				  -> Person Distributed Encoding
float w_pd2pv[N_DEP][N_PERSON];				//weights Person Distributed Encoding -> Person Vector(Decoding)

int iteration = 0;

inline float sigmoid(float z) { return (1 / (1 + exp(SLOPE_SIGMOID * -z)));}
inline float derivatives_sigmoid(float y) {return SLOPE_SIGMOID * y * (1 - y); }
//inline void idxToVec(int idx, float* vec, int n) {	for (int i = 0; i < n; i++)	vec[i] = idx == i ? 1.0f : 0.0f;}

bool isCorrect;

void InitData() {
	//w_pv2de
	for (int j = 0; j < N_DEP; j++)
	for (int i = 0; i < N_PERSON; i++)
		w_pv2de[i][j] = (int)((((rand() % 601) / 1000.0f) - 0.3) * 10) / 10.0f;

	//w_rv2de
	for (int j = N_DEP; j < N_DEP + N_DER; j++)
	for (int i = 0; i < N_RELATION; i++)
		w_rv2de[i][j - N_DEP] = (int)((((rand() % 601) / 1000.0f) - 0.3) * 10) / 10.0f;

	//w_de2cl
	for (int k = 0; k < N_CENTRAL; k++)
	for (int j = 0; j < N_DEP + N_DER; j++)
		w_de2cl[j][k] = (int)((((rand() % 601) / 1000.0f) - 0.3) * 10) / 10.0f;

	//w_cl2pd
	for (int l = 0; l < N_DEP; l++)
	for (int k = 0; k < N_CENTRAL; k++)
		w_cl2pd[k][l] = (int)((((rand() % 601) / 1000.0f) - 0.3) * 10) / 10.0f;

	//w_pd2pv
	for (int m = 0; m < N_PERSON; m++)
	for (int l = 0; l < N_DEP; l++)
		w_pd2pv[l][m] = (int)((((rand() % 601) / 1000.0f) - 0.3) * 10) / 10.0f;
}

float maxAverage = 0;
float minAverage = 0;

float FeedForward(int p1, int r, int p2, int c) {
	//Person Vector				    -> Person Distributed Encoding(Encoding)
	for (int j = 0; j < N_DEP; j++) 
		inputDE[c][j] = sigmoid(w_pv2de[p1][j]);

	//Relation Vector               -> Relation Distributed Encoding(Encoding)
	for (int j = 0; j < N_DER; j++) 
		inputDE[c][j + N_DEP] = sigmoid(w_rv2de[r][j]);

	//Distributed Encoding -> Central layer
	for (int k = 0; k < N_CENTRAL; k++) {
		float z_centralLayer = 0;
		for (int j = 0; j < N_DEP + N_DER; j++)	z_centralLayer += w_de2cl[j][k] * inputDE[c][j];
		centralLayer[c][k] = sigmoid(z_centralLayer);
	}

	//Central layer					-> Person Distributed Encoding
	for (int l = 0; l < N_DEP; l++) {
		float z_outputPersonDE = 0;
		for (int k = 0; k < N_CENTRAL; k++)	z_outputPersonDE += w_cl2pd[k][l] * centralLayer[c][k];
		outputPersonDE[c][l] = sigmoid(z_outputPersonDE);
	}

	//Person Distributed Encoding	-> Person Vector(Decoding)
	float error = 0;
	int isBiggerThanHalf = 0;
	float max = 0;
	float min = 100000000000;
	int maxIdx = -1;
	for (int m = 0; m < N_PERSON; m++) {
		float z_outputPersonVec = 0;
		for (int l = 0; l < N_DEP; l++)	z_outputPersonVec += w_pd2pv[l][m] * outputPersonDE[c][l];
		outputPersonVec[c][m] = sigmoid(z_outputPersonVec);
		//Error: { sum ( y - d )^2 } / 2
		{
			if (outputPersonVec[c][m] > max) 
				max = outputPersonVec[c][m]; maxIdx = m;
			if (outputPersonVec[c][m] < min) min = outputPersonVec[c][m];
			if (p2 == m)	error += pow(outputPersonVec[c][m] - 1, 2);
			else			error += pow(outputPersonVec[c][m] - 0, 2);
			if (p2 == m && outputPersonVec[c][m] >= 0.5)		isBiggerThanHalf = 1;
			else if (p2 == m && outputPersonVec[c][m] < 0.5)	isBiggerThanHalf--;
			else if (p2 != m && outputPersonVec[c][m] >= 0.5)	isBiggerThanHalf--;
		}
	}
	maxAverage += max;
	minAverage += min;
	isCorrect = isBiggerThanHalf == 1;
	return error / 2.0f;
}

float p_d_w_pv2de[N_PERSON][N_DEP];			//delta of weights Person Vector			   -> Person Distributed Encoding(Encoding)
float p_d_w_rv2de[N_RELATION][N_DER];			//delta of weights Relation Vector             -> Relation Distributed Encoding(Encoding)
float p_d_w_de2cl[N_DEP + N_DER][N_CENTRAL];	//delta of weights Distributed Encoding		   -> Central layer
float p_d_w_cl2pd[N_CENTRAL][N_DEP];			//delta of weights Central layer			   -> Person Distributed Encoding
float p_d_w_pd2pv[N_DEP][N_PERSON];			//delta of weights Person Distributed Encoding -> Person Vector(Decoding)

void BackPropagation(bool isFirst) {
	float d_w_pv2de[N_PERSON][N_DEP];			//delta of weights Person Vector			   -> Person Distributed Encoding(Encoding)
	float d_w_rv2de[N_RELATION][N_DER];			//delta of weights Relation Vector             -> Relation Distributed Encoding(Encoding)
	float d_w_de2cl[N_DEP + N_DER][N_CENTRAL];	//delta of weights Distributed Encoding		   -> Central layer
	float d_w_cl2pd[N_CENTRAL][N_DEP];			//delta of weights Central layer			   -> Person Distributed Encoding
	float d_w_pd2pv[N_DEP][N_PERSON];			//delta of weights Person Distributed Encoding -> Person Vector(Decoding)

	//Initialize deltas of weights
	//w_pv2de
	for (int j = 0; j < N_DEP; j++)
	for (int i = 0; i < N_PERSON; i++) {
		d_w_pv2de[i][j] = 0;
		if (isFirst) p_d_w_pv2de[i][j] = 0;
	}

	//w_rv2de
	for (int j = 0; j < N_DER; j++)
	for (int i = 0; i < N_RELATION; i++) {
		d_w_rv2de[i][j] = 0;
		if(isFirst) p_d_w_rv2de[i][j] = 0;
	}

	//w_de2cl
	for (int k = 0; k < N_CENTRAL; k++)
		for (int j = 0; j < N_DEP + N_DER; j++) {
			d_w_de2cl[j][k] = 0;
			if (isFirst) p_d_w_de2cl[j][k] = 0;
		}

	//w_cl2pd
	for (int l = 0; l < N_DEP; l++)
		for (int k = 0; k < N_CENTRAL; k++) {
			d_w_cl2pd[k][l] = 0;
			if (isFirst) p_d_w_cl2pd[k][l] = 0;
		}

	//w_pd2pv
	for (int m = 0; m < N_PERSON; m++)
		for (int l = 0; l < N_DEP; l++) {
			d_w_pd2pv[l][m] = 0;
			if (isFirst) p_d_w_pd2pv[l][m] = 0;
		}

	for (int c = 0; c < N_TRAIN; c++) {
		float pEBypZm[N_PERSON];
		float pEBypZl[N_DEP];
		float pEBypZk[N_CENTRAL];
		float pEBypZj[12];
		//Output => P2 Destributed Encoding: d_w_pd2pv
		for (int l = 0; l < N_DEP; l++) {			//l
			for (int m = 0; m < N_PERSON; m++) {	//m
				//ym -dm
				float pEBypYm = trainSet[c][2] == m ? outputPersonVec[c][m] - 1 : outputPersonVec[c][m];
				pEBypZm[m] = pEBypYm * derivatives_sigmoid(outputPersonVec[c][m]);
				d_w_pd2pv[l][m] += pEBypZm[m] * outputPersonDE[c][l]; //accumulate
			}
		}
		
		
		//P2 Distributed Encoding => Central: d_w_cl2pd
		for (int l = 0; l < N_DEP; l++) {
			pEBypZl[l] = 0;
			for (int m = 0; m < N_PERSON; m++)
				pEBypZl[l] += pEBypZm[m] * w_pd2pv[l][m];
			pEBypZl[l] = pEBypZl[l] * derivatives_sigmoid(outputPersonDE[c][l]);
		}
		
		for (int k = 0; k < N_CENTRAL; k++)
		for (int l = 0; l < N_DEP; l++)
			d_w_cl2pd[k][l] += pEBypZl[l] * centralLayer[c][k];

		//Central => Distributed Encoding: d_w_de2cl
		for (int k = 0; k < N_CENTRAL; k++) {
			pEBypZk[k] = 0;
			for (int l = 0; l < N_DEP; l++)
				pEBypZk[k] += pEBypZl[l] * w_cl2pd[k][l];
			pEBypZk[k] *= derivatives_sigmoid(centralLayer[c][k]);
		}
		for (int j = 0; j < N_DEP + N_DER; j++)
		for (int k = 0; k < N_CENTRAL; k++)
			d_w_de2cl[j][k] += pEBypZk[k] * inputDE[c][j];

		//Distributed Encoding => Relation Local: d_w_rv2de
		for (int j = N_DEP; j < N_DEP + N_DER; j++) {
			pEBypZj[j] = 0;
			for (int k = 0; k < N_CENTRAL; k++)
				pEBypZj[j] += pEBypZk[k] * w_de2cl[j][k];
			pEBypZj[j] *= derivatives_sigmoid(inputDE[c][j]);
		}

		for (int i = 0; i < N_RELATION; i++)
		for (int j = N_DEP; j < N_DEP + N_DER; j++)
			if (i == trainSet[c][1])
				d_w_rv2de[i][j - N_DEP] += pEBypZj[j];// *inputRelationVec[c][i];

		//Distributed Encoding => Person Local: d_w_pv2de
		for (int j = 0; j < N_DEP; j++) {
			pEBypZj[j] = 0;
			for (int k = 0; k < N_CENTRAL; k++)
				pEBypZj[j] += pEBypZk[k] * w_de2cl[j][k];
			pEBypZj[j] *= derivatives_sigmoid(inputDE[c][j]);
		}

		for (int i = 0; i < N_PERSON; i++)
		for (int j = 0; j < N_DEP; j++)
			if (i == trainSet[c][0])
				d_w_pv2de[i][j] += pEBypZj[j];// *inputPersonVec[c][i];
	}//training iteration finished

	//Update weights
	//w_pv2de
	for (int j = 0; j < N_DEP; j++)
		for (int i = 0; i < N_PERSON; i++) {
			w_pv2de[i][j] += -LR * d_w_pv2de[i][j] +ALPHA * p_d_w_pv2de[i][j];
			p_d_w_pv2de[i][j] = -LR * d_w_pv2de[i][j] +ALPHA * p_d_w_pv2de[i][j];
		}

	//w_rv2de
	for (int j = 0; j < N_DER; j++)
		for (int i = 0; i < N_RELATION; i++) {
			w_rv2de[i][j] += -LR * d_w_rv2de[i][j] +ALPHA * p_d_w_rv2de[i][j];
			p_d_w_rv2de[i][j] = -LR * d_w_rv2de[i][j] +ALPHA * p_d_w_rv2de[i][j];
		}

	//w_de2cl
	for (int k = 0; k < N_CENTRAL; k++)
		for (int j = 0; j < N_DEP + N_DER; j++) {
			w_de2cl[j][k] += -LR * d_w_de2cl[j][k] +ALPHA * p_d_w_de2cl[j][k];
			p_d_w_de2cl[j][k] = -LR * d_w_de2cl[j][k] +ALPHA * p_d_w_de2cl[j][k];
		}

	//w_cl2pd
	for (int l = 0; l < N_DEP; l++)
		for (int k = 0; k < N_CENTRAL; k++) {
			w_cl2pd[k][l] += -LR * d_w_cl2pd[k][l] +ALPHA * p_d_w_cl2pd[k][l];
			p_d_w_cl2pd[k][l] = -LR * d_w_cl2pd[k][l] +ALPHA * p_d_w_cl2pd[k][l];
		}
			

	//w_pd2pv
	for (int m = 0; m < N_PERSON; m++)
		for (int l = 0; l < N_DEP; l++) {
			w_pd2pv[l][m] += -LR * d_w_pd2pv[l][m] +ALPHA * p_d_w_pd2pv[l][m];
			p_d_w_pd2pv[l][m] = -LR * d_w_pd2pv[l][m] +ALPHA * p_d_w_pd2pv[l][m];
		}
}

int maxCorrectN = 0;
int maxIteration = 0;
int main() {
	//Local variables
	float E;
	float prevError = -1;
	

	//Load Train, Valid Set
	FILE *p = fopen("2016-30729_NN_Hinton_FamilyTree_submit.txt", "r");
	if (p == NULL) {
		printf("!!ERROR: No input file.");
		return 1;
	}

	char temp[3];
	for (int i = 0; i < 104; i++) {
		if (i < 100) {
			fscanf(p, "%s", temp);			trainSet[i][0] = atoi(temp);
			fscanf(p, "%s", temp);			trainSet[i][1] = atoi(temp);
			fscanf(p, "%s", temp);			trainSet[i][2] = atoi(temp);
		}else {
			fscanf(p, "%s", temp);			validSet[i-100][0] = atoi(temp);
			fscanf(p, "%s", temp);			validSet[i-100][1] = atoi(temp);
			fscanf(p, "%s", temp);			validSet[i-100][2] = atoi(temp);
		}
	}
	fclose(p);
	
	//Initialize weights
	InitData();

	//Training
	float prevE = 0;
	while(true) {
		E = 0.0f;

		int correctN = 0;
		maxAverage = 0;
		minAverage = 0;
		for (int c = 0; c < N_TRAIN;c++){
			E += FeedForward(trainSet[c][0], trainSet[c][1], trainSet[c][2], c);
			if (isCorrect) correctN++;
		}

		//printf("%.5f should be bigger than 0.5\n", maxAverage / N_TRAIN);
		//printf("%.5f should be smaller than 0.5\n", minAverage / N_TRAIN);
		int valCorrectN = 0;
		float VE = 0.0f;

		for (int c = 0; c < N_VALID; c++) {
			VE += FeedForward(validSet[c][0], validSet[c][1], validSet[c][2], c);
			if (isCorrect) {
				valCorrectN++;
			}
		}

		if(valCorrectN > maxCorrectN){
			maxIteration = iteration;
			maxCorrectN = valCorrectN;
		}
		printf("[%d] Error : %.5f, Learning Rate : %.5f, Decay Factor : %.5f, Train Correct %d out of %d, Valid Correct %d out of %d, Valid E %.5f\n", iteration, E, LR, ALPHA, correctN, N_TRAIN, valCorrectN, N_VALID, VE);

		if (iteration >= MAX_ITER) break;

		BackPropagation(!iteration); // iteration

		iteration++;

		if (VE < 1.5) {
			LR = 0.2;
			ALPHA = 0.35;
		}
		prevE = VE;
		if (VE < min_VE) {
			min_VE = VE;
			for (int i = 0; i < N_PERSON;i++)
			for (int j = 0; j < N_DEP; j++) min_w_pv2de[i][j] = w_pv2de[i][j];
			for (int i = 0; i < N_RELATION; i++)
			for (int j = 0; j < N_DER; j++) min_w_rv2de[i][j] = w_rv2de[i][j];
		}
	}
	printf("Max correctN is %d out of %d, error %.5f at %d\n", maxCorrectN, N_VALID, min_VE, maxIteration);

	//Print out the elements of second layer to reproduce results of the Hinton's paper
	//People
	printf("Weights from the 24 input units for people at %d\n", maxIteration);
	printf("%s %s %s %s %s %s %s %s %s %s %s %s\t%s %s %s %s %s %s %s %s %s %s %s %s\n",
		names[0], names[2], names[5], names[16], names[23], names[6], names[1], names[3], names[15], names[22], names[4], names[17],
		names[0], names[2], names[5], names[16], names[23], names[6], names[1], names[3], names[15], names[22], names[4], names[17]);
	printf("%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\t%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n"
	  , min_w_pv2de[0][3], min_w_pv2de[2][3], min_w_pv2de[5][3], min_w_pv2de[16][3], min_w_pv2de[23][3], min_w_pv2de[6][3], min_w_pv2de[1][3], min_w_pv2de[3][3], min_w_pv2de[15][3], min_w_pv2de[22][3], min_w_pv2de[4][3], min_w_pv2de[17][3],
		min_w_pv2de[0][0], min_w_pv2de[2][0], min_w_pv2de[5][0], min_w_pv2de[16][0], min_w_pv2de[23][0], min_w_pv2de[6][0], min_w_pv2de[1][0], min_w_pv2de[3][0], min_w_pv2de[15][0], min_w_pv2de[22][0], min_w_pv2de[4][0], min_w_pv2de[17][0]);
	printf("%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\t%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n"
	  , min_w_pv2de[7][3], min_w_pv2de[18][3], min_w_pv2de[10][3], min_w_pv2de[13][3], min_w_pv2de[21][3], min_w_pv2de[11][3], min_w_pv2de[8][3], min_w_pv2de[19][3], min_w_pv2de[12][3], min_w_pv2de[20][3], min_w_pv2de[9][3], min_w_pv2de[14][3],
		min_w_pv2de[7][0], min_w_pv2de[18][0], min_w_pv2de[10][0], min_w_pv2de[13][0], min_w_pv2de[21][0], min_w_pv2de[11][0], min_w_pv2de[8][0], min_w_pv2de[19][0], min_w_pv2de[12][0], min_w_pv2de[20][0], min_w_pv2de[9][0], min_w_pv2de[14][0]);
	printf("\n");
	printf("%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\t%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n"
	  , min_w_pv2de[0][4], min_w_pv2de[2][4], min_w_pv2de[5][4], min_w_pv2de[16][4], min_w_pv2de[23][4], min_w_pv2de[6][4], min_w_pv2de[1][4], min_w_pv2de[3][4], min_w_pv2de[15][4], min_w_pv2de[22][4], min_w_pv2de[4][4], min_w_pv2de[17][4],
		min_w_pv2de[0][1], min_w_pv2de[2][1], min_w_pv2de[5][1], min_w_pv2de[16][1], min_w_pv2de[23][1], min_w_pv2de[6][1], min_w_pv2de[1][1], min_w_pv2de[3][1], min_w_pv2de[15][1], min_w_pv2de[22][1], min_w_pv2de[4][1], min_w_pv2de[17][1]);
	printf("%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\t%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n"
	  , min_w_pv2de[7][4], min_w_pv2de[18][4], min_w_pv2de[10][3], min_w_pv2de[13][4], min_w_pv2de[21][4], min_w_pv2de[11][4], min_w_pv2de[8][4], min_w_pv2de[19][4], min_w_pv2de[12][4], min_w_pv2de[20][4], min_w_pv2de[9][4], min_w_pv2de[14][4],
		min_w_pv2de[7][1], min_w_pv2de[18][1], min_w_pv2de[10][3], min_w_pv2de[13][1], min_w_pv2de[21][1], min_w_pv2de[11][1], min_w_pv2de[8][1], min_w_pv2de[19][1], min_w_pv2de[12][1], min_w_pv2de[20][1], min_w_pv2de[9][1], min_w_pv2de[14][1]);
	printf("\n");
	printf("%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\t%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n", min_w_pv2de[0][5], min_w_pv2de[2][5], min_w_pv2de[5][5], min_w_pv2de[16][5], min_w_pv2de[23][5], min_w_pv2de[6][5], min_w_pv2de[1][5], min_w_pv2de[3][5], min_w_pv2de[15][5], min_w_pv2de[22][5], min_w_pv2de[4][5], min_w_pv2de[17][5],
		min_w_pv2de[0][2], min_w_pv2de[2][2], min_w_pv2de[5][2], min_w_pv2de[16][2], min_w_pv2de[23][2], min_w_pv2de[6][2], min_w_pv2de[1][2], min_w_pv2de[3][2], min_w_pv2de[15][2], min_w_pv2de[22][2], min_w_pv2de[4][2], min_w_pv2de[17][2]);
	printf("%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\t%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n"
		, min_w_pv2de[7][5], min_w_pv2de[18][5], min_w_pv2de[10][5], min_w_pv2de[13][5], min_w_pv2de[21][5], min_w_pv2de[11][5], min_w_pv2de[8][5], min_w_pv2de[19][5], min_w_pv2de[12][5], min_w_pv2de[20][5], min_w_pv2de[9][5], min_w_pv2de[14][5],
		min_w_pv2de[7][2], min_w_pv2de[18][2], min_w_pv2de[10][2], min_w_pv2de[13][2], min_w_pv2de[21][2], min_w_pv2de[11][2], min_w_pv2de[8][2], min_w_pv2de[19][2], min_w_pv2de[12][2], min_w_pv2de[20][2], min_w_pv2de[9][2], min_w_pv2de[14][2]);
	printf("%s %s %s %s %s %s %s %s %s %s %s %s\t%s %s %s %s %s %s %s %s %s %s %s %s\n",
		names[7], names[18], names[10], names[13], names[21], names[11], names[8], names[19], names[12], names[20], names[9], names[14],
		names[7], names[18], names[10], names[13], names[21], names[11], names[8], names[19], names[12], names[20], names[9], names[14]);
	printf("\n");

	printf("Weights from the 12 input units for relations at %d\n", maxIteration);
	printf("\n%s %s %s %s %s %s\t%s %s %s %s %s %s\t%s %s %s %s %s %s\n",
		relation[8], relation[11], relation[0], relation[1], relation[4], relation[5], relation[8], relation[11], relation[0], relation[1], relation[4], relation[5], relation[8], relation[11], relation[0], relation[1], relation[4], relation[5]);
	printf("%.2f %.2f %.2f %.2f %.2f %.2f\t%.2f %.2f %.2f %.2f %.2f %.2f\t%.2f %.2f %.2f %.2f %.2f %.2f\n",
		min_w_rv2de[8][4], min_w_rv2de[11][4], min_w_rv2de[0][4], min_w_rv2de[1][4], min_w_rv2de[4][4], min_w_rv2de[5][4],
		min_w_rv2de[8][2], min_w_rv2de[11][2], min_w_rv2de[0][2], min_w_rv2de[1][2], min_w_rv2de[4][2], min_w_rv2de[5][2],
		min_w_rv2de[8][0], min_w_rv2de[11][0], min_w_rv2de[0][0], min_w_rv2de[1][0], min_w_rv2de[4][0], min_w_rv2de[5][0]);

	printf("%.2f %.2f %.2f %.2f %.2f %.2f\t%.2f %.2f %.2f %.2f %.2f %.2f\t%.2f %.2f %.2f %.2f %.2f %.2f\n"
		, min_w_rv2de[10][4], min_w_rv2de[9][4], min_w_rv2de[2][4], min_w_rv2de[3][4], min_w_rv2de[6][4], min_w_rv2de[7][4],
		min_w_rv2de[10][2], min_w_rv2de[9][2], min_w_rv2de[2][2], min_w_rv2de[3][2], min_w_rv2de[6][2], min_w_rv2de[7][2],
		min_w_rv2de[10][0], min_w_rv2de[9][0], min_w_rv2de[2][0], min_w_rv2de[3][0], min_w_rv2de[6][0], min_w_rv2de[7][0]);
	printf("\n");
	printf("%.2f %.2f %.2f %.2f %.2f %.2f\t%.2f %.2f %.2f %.2f %.2f %.2f\t%.2f %.2f %.2f %.2f %.2f %.2f\n",
		min_w_rv2de[8][5], min_w_rv2de[11][5], min_w_rv2de[0][5], min_w_rv2de[1][5], min_w_rv2de[4][5], min_w_rv2de[5][5],
		min_w_rv2de[8][3], min_w_rv2de[11][3], min_w_rv2de[0][3], min_w_rv2de[1][3], min_w_rv2de[4][3], min_w_rv2de[5][3],
		min_w_rv2de[8][1], min_w_rv2de[11][1], min_w_rv2de[0][1], min_w_rv2de[1][1], min_w_rv2de[4][1], min_w_rv2de[5][1]);

	printf("%.2f %.2f %.2f %.2f %.2f %.2f\t%.2f %.2f %.2f %.2f %.2f %.2f\t%.2f %.2f %.2f %.2f %.2f %.2f\n"
		, min_w_rv2de[10][5], min_w_rv2de[9][5], min_w_rv2de[2][5], min_w_rv2de[3][5], min_w_rv2de[6][5], min_w_rv2de[7][5],
		min_w_rv2de[10][3], min_w_rv2de[9][3], min_w_rv2de[2][3], min_w_rv2de[3][3], min_w_rv2de[6][3], min_w_rv2de[7][3],
		min_w_rv2de[10][1], min_w_rv2de[9][1], min_w_rv2de[2][1], min_w_rv2de[3][1], min_w_rv2de[6][1], min_w_rv2de[7][1]);
	printf("%s %s %s %s %s %s\t%s %s %s %s %s %s\t%s %s %s %s %s %s\n",
		relation[10], relation[9], relation[2], relation[3], relation[6], relation[7], relation[10], relation[9], relation[2], relation[3], relation[6], relation[7], relation[10], relation[9], relation[2], relation[3], relation[6], relation[7]);
}
