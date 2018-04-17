#ifndef NODE_H
#define NODE_H


#include<stdio.h>
#include<vector>
#include<list>
#include <opencv.hpp>

#include <time.h>
#include <math.h>
#include <cmath>
#include<numeric>

using namespace std;
using namespace cv;

class Node{
private:
	//split parameter of node
	int x1;
	int x2;
	int y1;
	int y2;
	int d;
	float theta;
	int voting;

	//parameter of model
	int maxDepth;
	int minLeafSample;
	int minInfoGain;

	//status of node
	bool LeafFlag;
	int sample_num;
	int positive_num;
	int current_depth;
	float infoGain;
	float Entro;

	//data
	vector<Mat> imgList;
	vector<int> imgLabel;
	Node *leftchild;
	Node *rightchild;
	
public:
	Node(vector<Mat> &sample, vector<int> &label, int curr_depth, int w_w, int maxD, int minL, float minInfo);
	~Node();

	void setLeaf();
	inline bool isLeaf(){return LeafFlag;};

	float calculate_entropy(int sample_num, int positive_num);
	inline float get_infoGain(){return infoGain;};

	void train();
	void split_Node();

	int predict(Mat &test_img);
	//vector<int> predict(vector<Mat> &test_img);
};
#endif//NODE_H