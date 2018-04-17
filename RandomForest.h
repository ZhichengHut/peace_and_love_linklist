#ifndef RANDOMFOREST_H
#define RANDOMFOREST_H

#include <fstream>

#include "Node.h"

class RandomForest{
private:
	int window_width;
	int tree_num;
	int sample_num;
	int maxDepth;
	int minLeafSample;
	float minInfoGain;

	vector<Mat> imgData;
	vector<int> LabelData;

	Node **root_list;

public:
	RandomForest(vector<Mat> &img, vector<int> &label, int w_w, int t_n, int s_n, int maxD, int minL, float minInfo);
	~RandomForest();

	void train();
	
	float predict(Mat test_img);
	vector<float> predict(vector<Mat> &test_img);
};

#endif//RANDOMFOREST_H