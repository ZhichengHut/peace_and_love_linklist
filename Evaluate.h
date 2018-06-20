#ifndef EVALUATE_H
#define EVALUATE_H

#include "RandomForest.h"
#include "ExtractData.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>

#include<math.h>

void classifier(RandomForest *RF, string test_fold, int width);

void get_predict_result(RandomForest *RF, string test_fold);
void get_predict_result(RandomForest *RF, string test_fold, int width, int sample_interval, float prob_threshold);
void get_predict_result(RandomForest *RF, string test_fold, int width);

float get_F1_score(string test_fold);

bool TLBO_test(Mat &img, Mat &mask, float threshold);

Mat NMS(Mat &img_tmp, int window_width);

#endif//EVALUATE_H