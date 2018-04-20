#include "ExtractData.h"
#include "ReadData.h"
#include "Node.h"
#include "Data.h"
#include "RandomForest.h"
#include "Evaluate.h"

#include <time.h>

int main(){
	string train_fold = "C:/45 Thesis/data/train/";
	string test_fold = "C:/45 Thesis/data/test/";
	string out_fold = "C:/45 Thesis/data/train/extracted/";
	string out_fold_second = "C:/45 Thesis/data/train/extracted_second/";

	bool first_filter = false;
	bool second_filter = true;

	/*Mat haha = imread("C:/45 Thesis/data/train/none/4830_0.png",0);
	imshow("haha" ,haha);
	waitKey(0);
	cout << "mean = " << mean(haha(Rect(15,15,12,12))) << endl;
	integral(haha, haha);
	cout << "new mean = " << (haha.at<int>(27,27) +  haha.at<int>(15,15) - haha.at<int>(27,15) - haha.at<int>(15,27)) / 144.0<< endl;
	cin.get();*/

	if(first_filter){
		cout << "*****************Start to extract sub-image*****************" << endl;
		float train_thresh = 0.40;
		float test_thresh = 0.30;

		bool get_train = false;
		bool get_test = false;

		int patch_width = 35;
		int core_R = 4;
		int ran_point = 40;

		extractData(train_fold, test_fold, out_fold, train_thresh, test_thresh, get_train, get_test, patch_width, core_R, ran_point);
		cout << "*****************Extraction completed*****************" << endl << endl;

		cout << "*****************Start to read training data*****************" << endl;
		vector<Mat> imgTrain;
		vector<Mat> integral_img_list;
		vector<int> labelTrain;

		readTrainData(out_fold, integral_img_list, labelTrain);

		cout << "Sample number = " << integral_img_list.size() << endl;
		cout << "Positive sampe number = " << accumulate(labelTrain.begin(), labelTrain.end(),0) << endl;
		cout << "*****************Reading completed*****************" << endl << endl;

		double start,end,cost;

		for(float i=1; i<=1; i++){
			int window_width = i;

			int tree_num = 30;
			int sample_num = 10000;
			int maxDepth = 50;
			int minLeafSample = 50;
			float minInfo = 0;

			cout << "*****************Start to train the model*****************" << endl;
			start=clock();
			RandomForest *RF = new RandomForest(integral_img_list, labelTrain, window_width, tree_num, sample_num, maxDepth, minLeafSample, minInfo);
			RF->train();
			end = clock();
			double train_t = (end - start) / CLOCKS_PER_SEC ;
			cout << "*****************Training completed*****************" << endl << endl;

			cout << "*****************Start to evaluate the performance*****************" << endl;
			start=clock();
			//get_predict_result(RF, test_fold);
			int sample_interval = 5;
			float prob_threshold = 0.4;
			get_predict_result(RF, test_fold, patch_width, sample_interval, prob_threshold);
			//get_predict_result(RF, test_fold);
			end=clock();
			double test_t = (end - start) / CLOCKS_PER_SEC ;
			cout << "*****************Evaluation completed*****************" << endl << endl;

			cout << "*****************Start to calculate F1 score*****************" << endl;
			float F1_score = get_F1_score(test_fold);
			cout << "*****************Calculation completed*****************" << endl << endl;

			ofstream fin("e:\\45 Thesis\\result\\result.csv",ios::app);
			if(!fin){
				cout << "open file error" <<endl; 
				cin.get();
				return 0;
			}

			fin <<",tree num," <<  tree_num << ",sumple num," << sample_num << ",maxDepth," << maxDepth << ",minLeafSample," << minLeafSample << ",minInfo," << minInfo <<",train time," << train_t << ",test time," << test_t <<",window width," << window_width << endl;;
			fin.close();

			delete RF;
			RF = NULL;
		}
		imgTrain.clear();
		vector<Mat>().swap(imgTrain); 
		integral_img_list.clear();
		vector<Mat>().swap(integral_img_list);
		labelTrain.clear();
		vector<int>().swap(labelTrain);
	}

	if(second_filter){
		cout << "*****************Start to extract sub-image*****************" << endl;
		float train_thresh = 0.4;
		bool get_train = false;

		int patch_width = 35;
		int core_R = 4;
		int ran_point = 40;

		extractData(train_fold, out_fold_second, train_thresh,get_train, patch_width, core_R);
		cout << "*****************Extraction completed*****************" << endl << endl;

		cout << "*****************Start to read training data*****************" << endl;
		vector<Mat> imgTrain;
		vector<Mat> integral_img_list;
		vector<int> labelTrain;

		readTrainData(out_fold_second, integral_img_list, labelTrain);

		cout << "Sample number = " << integral_img_list.size() << endl;
		cout << "Positive sampe number = " << accumulate(labelTrain.begin(), labelTrain.end(),0) << endl;
		cout << "*****************Reading completed*****************" << endl << endl;

		double start,end,cost;

		for(float i=1; i<=1; i++){
			int window_width = i;

			int tree_num = 30;
			int sample_num = 10000;
			int maxDepth = 50;
			int minLeafSample = 50;
			float minInfo = 0;

			cout << "*****************Start to train the model*****************" << endl;
			start=clock();
			RandomForest *RF = new RandomForest(integral_img_list, labelTrain, window_width, tree_num, sample_num, maxDepth, minLeafSample, minInfo);
			RF->train();
			end = clock();
			double train_t = (end - start) / CLOCKS_PER_SEC ;
			cout << "*****************Training completed*****************" << endl << endl;

			cout << "*****************Start to evaluate the performance*****************" << endl;
			start=clock();
			get_predict_result(RF, test_fold, patch_width);
			end=clock();
			double test_t = (end - start) / CLOCKS_PER_SEC ;
			cout << "*****************Evaluation completed*****************" << endl << endl;

			cout << "*****************Start to calculate F1 score*****************" << endl;
			float F1_score = get_F1_score(test_fold, second_filter);
			cout << "*****************Calculation completed*****************" << endl << endl;

			ofstream fin("e:\\45 Thesis\\result\\result.csv",ios::app);
			if(!fin){
				cout << "open file error" <<endl; 
				cin.get();
				return 0;
			}

			fin <<",tree num," <<  tree_num << ",sumple num," << sample_num << ",maxDepth," << maxDepth << ",minLeafSample," << minLeafSample << ",minInfo," << minInfo <<",train time," << train_t << ",test time," << test_t <<",window width," << window_width << endl;;
			fin.close();

			delete RF;
			RF = NULL;
		}
		imgTrain.clear();
		vector<Mat>().swap(imgTrain); 
		integral_img_list.clear();
		vector<Mat>().swap(integral_img_list);
		labelTrain.clear();
		vector<int>().swap(labelTrain);
	}



	cout << "*****************Benchmark completed*****************" << endl;
	cin.get();
}