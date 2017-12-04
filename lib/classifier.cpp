#include "classifier.h" 

SegClassifier::SegClassifier(const string& model_file,
                             const string& trained_file,
                             /*const cv::Mat& matSrcMat,*/
                             const cv::Mat& matMeanImage,
                             int nCrop_Width,
                             int nCrop_Height
                             /*const string& label_file*/) {
//#ifdef CPU_ONLY
  //Caffe::set_mode(Caffe::CPU);
//#else
  Caffe::set_mode(Caffe::GPU);
//#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
  
  m_nCrop_Width = nCrop_Width;
  m_nCrop_Height = nCrop_Height;
  
  /* Load the binaryproto mean file. */
  //SetMean(mean_file);
  CHECK(matMeanImage.size() == input_geometry_)
    << "Mean image should be same size with input layer!";
  m_matMean = matMeanImage;
  m_bSegPerson = true;
  m_bSegMask = true;
  
  /*Initialize result image*/
  //m_matSrcMat = matSrcMat;
  
  //if(3 == num_channels_ )
    //m_matResultMat = cv::Mat::zeros(cv::Size(m_nCrop_Width, m_nCrop_Height), CV_8UC3);
  //else if(1 == num_channels_)
    //m_matResultMat = cv::Mat::zeros(cv::Size(m_nCrop_Width, m_nCrop_Height), CV_8UC1);

  /* Load labels. */
  
  //std::ifstream labels(label_file.c_str());
  //CHECK(labels) << "Unable to open labels file " << label_file;
  //string line;
  //while (std::getline(labels, line))
    //labels_.push_back(string(line));

  //Blob<float>* output_layer = net_->output_blobs()[0];
  //CHECK_EQ(labels_.size(), output_layer->channels())
    //<< "Number of labels is different from the output layer dimension.";
}

/* Return the top N predictions. */
void SegClassifier::Classify(Mat& matProcessMat) {
  Caffe::set_mode(Caffe::GPU);
  //cout << "Classify function run!" << endl;
  Predict(matProcessMat);
  //cout << "Classify function end!" << endl;
  
  /*CHECK(output.size() == m_matResultMat.rows * m_matResultMat.cols)
    << "Outlayer size is not same with result mat!";
    

  int nResultImageWidth = m_matResultMat.cols;
  int nResultImageHeight = m_matResultMat.rows;
  if(m_matResultMat.isContinuous ()){
    float* pResultImage = (float*)(m_matResultMat.data);
    for(int i=0; i < nResultImageHeight*nResultImageWidth; i++){
        pResultImage[i] = output[i];
    }
  }else{
    for(int j = 0; j < nResultImageHeight; j++){
        float *pResultPtr = m_matResultMat.ptr<float>(j);
        for(int i = 0; i < nResultImageWidth; i++){
            pResultPtr[i] = output[j*nResultImageWidth + i];
        }
    }
  }*/

}

void SegClassifier::PermuteImage(const Mat&img, const Mat&meanimg, Mat& preimg){
    if(img.empty()){
        cerr << "Permute input image is empty!" << endl;
        exit(-1);
    }
    if(3 != img.channels()){
        cerr << "Permute input image's channel number error!" << endl;
        exit(-1);
    }
    
    vector<Mat>vmatChannels;
    vector<Mat>vmatMeanImage;
    split(img, vmatChannels);
    split(meanimg, vmatMeanImage);
    Mat tempMat = vmatChannels[0];
    vmatChannels[0] = vmatChannels[2];
    vmatChannels[2] = tempMat;
    
    for(int n = 0; n < vmatChannels.size(); n++){
        vmatChannels[n] = vmatChannels[n] - vmatMeanImage[n]; 
    }
    
    merge(vmatChannels, preimg);
}


void SegClassifier::Predict(const cv::Mat& img) {
  //cout << "Predict function run 1!" << endl;
  Blob<float>* input_layer = net_->input_blobs()[0];
  Mat resizeimg;
  
  resize(img, resizeimg, input_geometry_);
  //cout << "Predict function run 2!" << endl;  
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();
  //cout << "Predict function run 3!" << endl;
  std::vector<cv::Mat> input_channels;
  double dWrapInputLayerCurrentTime  = (double)(getTickCount());
  WrapInputLayer(&input_channels);
  double dWrapInputLayerProcessTime = ((double)(getTickCount()) - dWrapInputLayerCurrentTime) / getTickFrequency();

  double dPreprocessCurrentTime  = (double)(getTickCount());
  Preprocess(resizeimg, &input_channels);
  double dPreprocessProcessTime = ((double)(getTickCount()) - dPreprocessCurrentTime) / getTickFrequency();
  //cout << "Predict function run 4!" << endl;
  double dProcessTime = (double)(getTickCount());
  float fLoss;
  net_->ForwardPrefilled(&fLoss);
  dProcessTime = ((double)(getTickCount()) - dProcessTime)/getTickFrequency();
  //cout << "The net forward time is:" << dProcessTime << endl;
  //cout << "Predict function run 5!" << endl;
  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
  //const float* begin = output_layer->cpu_data();
  //const float* end = begin + output_layer->channels() * output_layer->width() * output_layer->height();
  //return std::vector<float>(begin, end);
  double dObtainResultCurrentTime  = (double)(getTickCount());
  ObtainResult(output_layer, img);
  double dObtainResultProcessTime = ((double)(getTickCount()) - dObtainResultCurrentTime) / getTickFrequency();
  //cout << "Predict function run 6!" << endl;
  cout << "dWrapInputLayerProcessTime is:" << dWrapInputLayerProcessTime << endl;
  cout << "dPreprocessProcessTime is:" << dPreprocessProcessTime << endl; 
  cout << "dObtainResultProcessTime is:" << dObtainResultProcessTime << endl;
}

void SegClassifier::ObtainResult(Blob<float>* output_layer, const Mat& matSrcMat){
    //cout << "ObtainResult function 1!" << endl;
    float* pOutNet = output_layer->mutable_cpu_data();
    
    int nChannels = output_layer->channels();
    int nWidth = output_layer->width();
    int nHeight = output_layer->height();
    int nMatSize = nWidth * nHeight;
    
    /*if(!m_matGrayResultMat.empty())
        m_matGrayResultMat.release();
    if(!m_matColorResultMat.empty())
        m_matColorResultMat.release();    
    //cout << "ObtainResult function 3!" << endl;    
    m_matGrayResultMat = Mat(nHeight, nWidth, CV_8UC1);
    uchar* pResultMat = m_matGrayResultMat.data;
    m_matColorResultMat = Mat(nHeight, nWidth, CV_8UC3);
    Vec3b* pIndexMat = m_matColorResultMat.ptr<Vec3b>(0);
    
    double dFigureCurrentTime = (double)getTickCount();
    int nN, nM; 
    float fMaxResult;
    int m, n;
    int nWH = nWidth * nHeight;
    for(n = 0; n < nWH; n++){
        fMaxResult = pOutNet[n];
        pResultMat[n ] = 0;
        pIndexMat[n ] = pv3bColorMap[0];
        for(m = 1;  m < nChannels; m++){
            //nN = n%nChannels;
            //nM = n/nChannels;

            if(pOutNet[m * nWH + n] > fMaxResult){
                pResultMat[n ] = m;
                pIndexMat[n ] = pv3bColorMap[m];
                fMaxResult = pOutNet[m * nWH + n];
            }

        }

    
    }
    double dFigureProcessTime = ((double)getTickCount() - dFigureCurrentTime) / getTickFrequency(); 
    cout << "dFigureProcessTime is:" << dFigureProcessTime << endl;*/
    //cout << "OutLayer's channels is:" << nChannels << endl;
    //cout << "OutLayer's width is:" << nWidth << endl;
    //cout << "OutLayer's height is:" << nHeight << endl;
    
    double dFigureCurrentTime = (double)getTickCount();
    vector<Mat>vmatOutputLayer(nChannels);
    vector<float*> vpfOutputLayer(nChannels);
    int l, m, n;
    int nTempM;
    for(l = 0; l < nChannels; l++){
        vmatOutputLayer[l] = Mat(nHeight, nWidth, CV_32FC1);
        vpfOutputLayer[l] = (float*)(vmatOutputLayer[l].data);
   }
   //cout << "ObtainResult function 2!" << endl;
   
   float fTempSum, fTempMax;
   int nTempPos;
   for(m = 0; m < nHeight; m++){
        nTempM = m * nWidth;
        for(n = 0; n < nWidth; n++){
            fTempSum = 0.0f;
            for(l = 0; l < nChannels; l++){
                //vpfOutputLayer[l][nTempM + n] = (float)(exp(pOutNet [l * nMatSize + nTempM + n]));
                vpfOutputLayer[l][nTempM + n] = pOutNet [l * nMatSize + nTempM + n];
                //fTempSum += vpfOutputLayer[l][nTempM + n];
            }
            //for(l = 0; l < nChannels; l++){
                //vpfOutputLayer[l][nTempM + n] = vpfOutputLayer[l][nTempM + n] / fTempSum;
            //}
        }
    }
    double dFigureProcessTime = ((double)getTickCount() - dFigureCurrentTime) / getTickFrequency();
    double dExpCurrentTime = (double)getTickCount();
    exp(1);
    double dExpProcessTime = ((double)getTickCount() - dExpCurrentTime) / getTickFrequency();

    double dLabelCurrentTime = (double)getTickCount();
    for(l = 0; l < nChannels; l++){
        resize(vmatOutputLayer[l], vmatOutputLayer[l], Size(matSrcMat.cols, matSrcMat.rows));
        vpfOutputLayer[l]= (float *)(vmatOutputLayer[l].data);    
    }
    
    if(!m_matGrayResultMat.empty())
        m_matGrayResultMat.release();
    if(!m_matColorResultMat.empty())
        m_matColorResultMat.release();    
    //cout << "ObtainResult function 3!" << endl;    
    m_matGrayResultMat = Mat(matSrcMat.rows, matSrcMat.cols, CV_8UC1);
    uchar* pResultMat = m_matGrayResultMat.data;
    m_matColorResultMat = Mat(matSrcMat.rows, matSrcMat.cols, CV_8UC3);
    Vec3b* pIndexMat = m_matColorResultMat.ptr<Vec3b>(0);
    
    for(m = 0; m < matSrcMat.rows; m++){
        nTempM = m * matSrcMat.cols;
        for(n = 0; n < matSrcMat.cols; n++){
            fTempMax = 0.0f;
            for(l = 0; l < nChannels; l++){
                if(fTempMax < vpfOutputLayer[l][nTempM + n]){
                    fTempMax = vpfOutputLayer[l][nTempM + n];
                    pResultMat[nTempM + n] = l;
                }
            }
            if(m_bSegMask){
                if(15 != pResultMat[nTempM + n])
                    pResultMat[nTempM + n] = 0;
                else
                    pResultMat[nTempM + n] = 255;
                pIndexMat[nTempM + n] = Vec3b(255, 255, 255);
            }else{
                if(m_bSegPerson)
                    if(15 != pResultMat[nTempM + n])
                        pResultMat[nTempM + n] = 0;
                pIndexMat[nTempM + n] = pv3bColorMap[pResultMat[nTempM + n]];
            }

        }
    }
    double dLabelProcessTime = ((double)getTickCount() - dLabelCurrentTime) / getTickFrequency();
    
    cout << "dFigureProcessTime is:" << dFigureProcessTime << endl;
    cout << "dLabelProcessTime is:" << dLabelProcessTime << endl;
    cout << "dExpProcessTime is:" << dExpProcessTime << endl;
    //namedWindow("ColorResultImage");
    //imshow("ColorResultImage", m_matColorResultMat);
    //waitKey(1);
    //cout << "ObtainResult function 4!" << endl;    
    
}


/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void SegClassifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void SegClassifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  PermuteImage(sample_float, m_matMean, sample_normalized);

  //cv::subtract(sample_float, m_matMean, sample_normalized);


  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

  void SegClassifier::SetMean(const cv::Mat& matMeanImage){
      m_matMean = matMeanImage;
}

  cv::Mat SegClassifier::GetGrayResult(){
      return m_matGrayResultMat;
}

  cv::Mat SegClassifier::GetColorResult(){
      return m_matColorResultMat;
}

/* Load the mean file in binaryproto format. */
// void SegClassifier::SetMean(const string& mean_file) {
//   BlobProto blob_proto;
//   ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
// 
//   /* Convert from BlobProto to Blob<float> */
//   Blob<float> mean_blob;
//   mean_blob.FromProto(blob_proto);
//   CHECK_EQ(mean_blob.channels(), num_channels_)
//     << "Number of channels of mean file doesn't match input layer.";
// 
//   /* The format of the mean file is planar 32-bit float BGR or grayscale. */
//   std::vector<cv::Mat> channels;
//   float* data = mean_blob.mutable_cpu_data();
//   for (int i = 0; i < num_channels_; ++i) {
//     /* Extract an individual channel. */
//     cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
//     channels.push_back(channel);
//     data += mean_blob.height() * mean_blob.width();
//   }
// 
//   /* Merge the separate channels into a single image. */
//   cv::Mat mean;
//   cv::merge(channels, mean);
// 
//   /* Compute the global mean pixel value and create a mean image
//    * filled with this value. */
//   cv::Scalar channel_mean = cv::mean(mean);
//   mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
// }
