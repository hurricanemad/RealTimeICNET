#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <caffe/caffe.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <cmath>

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;
using namespace cv;
using namespace std;

class SegClassifier {
 public:
    SegClassifier(const string& ,
                  const string& ,
                  const cv::Mat& ,
                  int ,
                  int ) ; 

  
 void Classify(Mat& );

 private:
  //void SetMean(const string& mean_file);
  void SetMean(const cv::Mat& );

  void Predict(const cv::Mat& );

  void WrapInputLayer(std::vector<cv::Mat>* );

  void Preprocess(const cv::Mat& ,
                  std::vector<cv::Mat>* );
  void PermuteImage(const Mat&, const Mat&, Mat& );
  
  void ObtainResult(Blob<float>*, const Mat&);
  
public: 
  Mat GetGrayResult();
  Mat GetColorResult();
  
  
 private:
  std::shared_ptr< Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat m_matMean;
  int m_nCrop_Width; 
  int m_nCrop_Height;
  //std::vector<string> labels_;
  cv::Mat m_matGrayResultMat;
  cv::Mat m_matColorResultMat;
  cv::Mat m_matSrcMat;
  bool m_bSegPerson;
  bool m_bSegMask;
};

Vec3b pv3bColorMap[] = {Vec3b(0, 0,0),
                        Vec3b(0,0,128),
                        Vec3b(0,128,0),
                        Vec3b(0,128,128),
                        Vec3b(128,   0, 0),
                        Vec3b(128,   0, 128),
                        Vec3b(128, 128, 0),
                        Vec3b(128, 128, 128),
                        Vec3b( 0,   0,   64),
                        Vec3b( 0,   0,  192),
                        Vec3b( 0, 128,   64),
                        Vec3b( 0, 128,   192),
                        Vec3b(128,   0, 64),
                        Vec3b(128,   0, 192),
                        Vec3b(128, 128, 64),
                        Vec3b(128, 128, 192),
                        Vec3b(  0,  64,   0),
                        Vec3b(  0,  64,   128),
                        Vec3b(  0, 192,   0),
                        Vec3b(  0, 192,   128),
                        Vec3b(  128,  64, 0)};

/*cv::Vec3b pv3bColorMap[] = {
	Vec3b(129,64,129),
	Vec3b(233,35,245),
	Vec3b(70,70,70),
	Vec3b(157,102,102),
	Vec3b(154,154,191),
	Vec3b(154,154,154),
	Vec3b(30,171,251),
	Vec3b(0,221,221),
	Vec3b(35,143,107),
	Vec3b(153,252,153),
	Vec3b(181,131,70),
	Vec3b(60,20,221),
	Vec3b(0,0,255),
	Vec3b(143,0,0),
	Vec3b(70,0,0),
	Vec3b(100,60,0),
	Vec3b(100,80,0),
	Vec3b(231,0,0),
	Vec3b(32,11,119),
	Vec3b(0,193,129),
	Vec3b(129,64,0),
	Vec3b(129,64,129),
	Vec3b(129,193,0),
	Vec3b(129,193,129),
	Vec3b(0,64,64),
	Vec3b(0,64,193),
	Vec3b(0,193,64),
	Vec3b(0,193,193),
	Vec3b(129,64,64),
	Vec3b(129,64,193),
	Vec3b(129,193,64),
	Vec3b(129,193,193),
	Vec3b(64,0,0),
	Vec3b(64,0,129),
	Vec3b(64,129,0),
	Vec3b(64,129,129),
	Vec3b(193,0,0),
	Vec3b(193,0,129),
	Vec3b(193,129,0),
	Vec3b(193,129,129),
	Vec3b(64,0,64),
	Vec3b(64,0,193),
	Vec3b(64,129,64),
	Vec3b(64,129,193),
	Vec3b(193,0,64),
	Vec3b(193,0,193),
	Vec3b(193,129,64),
	Vec3b(193,129,193),
	Vec3b(64,64,0),
	Vec3b(64,64,129),
	Vec3b(64,193,0),
	Vec3b(64,193,129),
	Vec3b(193,64,0),
	Vec3b(193,64,129),
	Vec3b(193,193,0),
	Vec3b(193,193,129),
	Vec3b(64,64,64),
	Vec3b(64,64,193),
	Vec3b(64,193,64),
	Vec3b(64,193,193),
	Vec3b(193,64,64),
	Vec3b(193,64,193),
	Vec3b(193,193,64),
	Vec3b(193,193,193),
	Vec3b(0,0,32),
	Vec3b(0,0,161),
	Vec3b(0,129,32),
	Vec3b(0,129,161),
	Vec3b(129,0,32),
	Vec3b(129,0,161),
	Vec3b(129,129,32),
	Vec3b(129,129,161),
	Vec3b(0,0,96),
	Vec3b(0,0,225),
	Vec3b(0,129,96),
	Vec3b(0,129,225),
	Vec3b(129,0,96),
	Vec3b(129,0,225),
	Vec3b(129,129,96),
	Vec3b(129,129,225),
	Vec3b(0,64,32),
	Vec3b(0,64,161),
	Vec3b(0,193,32),
	Vec3b(0,193,161),
	Vec3b(129,64,32),
	Vec3b(129,64,161),
	Vec3b(129,193,32),
	Vec3b(129,193,161),
	Vec3b(0,64,96),
	Vec3b(0,64,225),
	Vec3b(0,193,96),
	Vec3b(0,193,225),
	Vec3b(129,64,96),
	Vec3b(129,64,225),
	Vec3b(129,193,96),
	Vec3b(129,193,225),
	Vec3b(64,0,32),
	Vec3b(64,0,161),
	Vec3b(64,129,32),
	Vec3b(64,129,161),
	Vec3b(193,0,32),
	Vec3b(193,0,161),
	Vec3b(193,129,32),
	Vec3b(193,129,161),
	Vec3b(64,0,96),
	Vec3b(64,0,225),
	Vec3b(64,129,96),
	Vec3b(64,129,225),
	Vec3b(193,0,96),
	Vec3b(193,0,225),
	Vec3b(193,129,96),
	Vec3b(193,129,225),
	Vec3b(64,64,32),
	Vec3b(64,64,161),
	Vec3b(64,193,32),
	Vec3b(64,193,161),
	Vec3b(193,64,32),
	Vec3b(193,64,161),
	Vec3b(193,193,32),
	Vec3b(193,193,161),
	Vec3b(64,64,96),
	Vec3b(64,64,225),
	Vec3b(64,193,96),
	Vec3b(64,193,225),
	Vec3b(193,64,96),
	Vec3b(193,64,225),
	Vec3b(193,193,96),
	Vec3b(193,193,225),
	Vec3b(0,32,0),
	Vec3b(0,32,129),
	Vec3b(0,161,0),
	Vec3b(0,161,129),
	Vec3b(129,32,0),
	Vec3b(129,32,129),
	Vec3b(129,161,0),
	Vec3b(129,161,129),
	Vec3b(0,32,64),
	Vec3b(0,32,193),
	Vec3b(0,161,64),
	Vec3b(0,161,193),
	Vec3b(129,32,64),
	Vec3b(129,32,193),
	Vec3b(129,161,64),
	Vec3b(129,161,193),
	Vec3b(0,96,0),
	Vec3b(0,96,129),
	Vec3b(0,225,0),
	Vec3b(0,225,129),
	Vec3b(129,96,0),
	Vec3b(129,96,129),
	Vec3b(129,225,0),
	Vec3b(129,225,129),
	Vec3b(0,96,64),
	Vec3b(0,96,193),
	Vec3b(0,225,64),
	Vec3b(0,225,193),
	Vec3b(129,96,64),
	Vec3b(129,96,193),
	Vec3b(129,225,64),
	Vec3b(129,225,193),
	Vec3b(64,32,0),
	Vec3b(64,32,129),
	Vec3b(64,161,0),
	Vec3b(64,161,129),
	Vec3b(193,32,0),
	Vec3b(193,32,129),
	Vec3b(193,161,0),
	Vec3b(193,161,129),
	Vec3b(64,32,64),
	Vec3b(64,32,193),
	Vec3b(64,161,64),
	Vec3b(64,161,193),
	Vec3b(193,32,64),
	Vec3b(193,32,193),
	Vec3b(193,161,64),
	Vec3b(193,161,193),
	Vec3b(64,96,0),
	Vec3b(64,96,129),
	Vec3b(64,225,0),
	Vec3b(64,225,129),
	Vec3b(193,96,0),
	Vec3b(193,96,129),
	Vec3b(193,225,0),
	Vec3b(193,225,129),
	Vec3b(64,96,64),
	Vec3b(64,96,193),
	Vec3b(64,225,64),
	Vec3b(64,225,193),
	Vec3b(193,96,64),
	Vec3b(193,96,193),
	Vec3b(193,225,64),
	Vec3b(193,225,193),
	Vec3b(0,32,32),
	Vec3b(0,32,161),
	Vec3b(0,161,32),
	Vec3b(0,161,161),
	Vec3b(129,32,32),
	Vec3b(129,32,161),
	Vec3b(129,161,32),
	Vec3b(129,161,161),
	Vec3b(0,32,96),
	Vec3b(0,32,225),
	Vec3b(0,161,96),
	Vec3b(0,161,225),
	Vec3b(129,32,96),
	Vec3b(129,32,225),
	Vec3b(129,161,96),
	Vec3b(129,161,225),
	Vec3b(0,96,32),
	Vec3b(0,96,161),
	Vec3b(0,225,32),
	Vec3b(0,225,161),
	Vec3b(129,96,32),
	Vec3b(129,96,161),
	Vec3b(129,225,32),
	Vec3b(129,225,161),
	Vec3b(0,96,96),
	Vec3b(0,96,225),
	Vec3b(0,225,96),
	Vec3b(0,225,225),
	Vec3b(129,96,96),
	Vec3b(129,96,225),
	Vec3b(129,225,96),
	Vec3b(129,225,225),
	Vec3b(64,32,32),
	Vec3b(64,32,161),
	Vec3b(64,161,32),
	Vec3b(64,161,161),
	Vec3b(193,32,32),
	Vec3b(193,32,161),
	Vec3b(193,161,32),
	Vec3b(193,161,161),
	Vec3b(64,32,96),
	Vec3b(64,32,225),
	Vec3b(64,161,96),
	Vec3b(64,161,225),
	Vec3b(193,32,96),
	Vec3b(193,32,225),
	Vec3b(193,161,96),
	Vec3b(193,161,225),
	Vec3b(64,96,32),
	Vec3b(64,96,161),
	Vec3b(64,225,32),
	Vec3b(64,225,161),
	Vec3b(193,96,32),
	Vec3b(193,96,161),
	Vec3b(193,225,32),
	Vec3b(193,225,161),
	Vec3b(64,96,96),
	Vec3b(64,96,225),
	Vec3b(64,225,96),
	Vec3b(64,225,225),
	Vec3b(193,96,96),
	Vec3b(193,96,225),
	Vec3b(193,225,96),
	Vec3b(0,0,0)
};*/

std::vector<double>vdMeanColorValue={123.68, 116.779, 103.939};


#endif
