#include "RandomForest.h"


RandomForest::RandomForest(vector<Mat> &img, vector<int> &label, int w_w, int t_n, int s_n, int maxD, int minL, float minInfo){
	window_width = w_w;

	//imgData.assign(img.begin(), img.end());
	//LabelData.assign(label.begin(), label.end());
	imgData = img;
	LabelData = label;
	//cout << "sum: = " << accumulate(LabelData.begin(),LabelData.end(), 0) << endl;

	tree_num = t_n;
	maxDepth = maxD;
	minLeafSample = minL;
	minInfoGain = minInfo;
	root_list = new Node*[tree_num];

	if(s_n > imgData.size()){
		cout << "Sample size out of range, " << imgData.size() << " sample will be used" << endl;
		sample_num = imgData.size();
	}
	else
		sample_num = s_n;
}

RandomForest::~RandomForest(){
	for(int i=0; i<tree_num; i++){
		if(root_list[i] != NULL){
			delete root_list[i];
			root_list[i] = NULL;
		}
	}

	delete[] root_list;
	root_list = NULL;

	imgData.clear();
	LabelData.clear();
}

void RandomForest::train(){
	srand(unsigned(time(NULL)));

	for(int i=0; i<tree_num; i++){
		cout << "Start to train the " << i << "th tree" << endl;
		
		float sp = 1.0*sample_num/imgData.size();

		vector<Mat> img;
		vector<int> lab;

		for(int i=0; i<imgData.size(); i++){
			if((rand()%10001)/10000.0<=sp){
				img.push_back(imgData[i]);
				lab.push_back(LabelData[i]);
			}
		}

		root_list[i] = new Node(img, lab, 0, window_width, maxDepth, minLeafSample, minInfoGain);
		img.clear();
		lab.clear();
		vector<Mat>().swap(img);
		vector<int>().swap(lab);
		root_list[i]->train();
	}
}

float RandomForest::predict(Mat test_img){
	int vote = 0;
	for(int j=0; j<tree_num; j++)
		vote += root_list[j]->predict(test_img);
	
	return 1.0*vote/tree_num;
}

vector<float> RandomForest::predict(vector<Mat> &test_img){
	//cout << "Start to predict" << endl;
	//cout << "test size = " << test_img.size() << endl;
	vector<float> predict_result;
	for(int i=0; i<test_img.size(); i++){
		predict_result.push_back(predict(test_img[i]));
	}
	//cout << "predict size = " << predict_result.size() << endl;

	return predict_result;
}