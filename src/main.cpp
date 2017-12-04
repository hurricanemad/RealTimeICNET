#include "prefix.hpp"

pthread_mutex_t ptmMutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t ptmCond = PTHREAD_COND_INITIALIZER;

pthread_mutex_t ptmSegMutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t ptmSegCond = PTHREAD_COND_INITIALIZER;

pthread_t pthreadProducter, pthreadConsumer, pthreadSegmentation;

void* ProductFunc(void * pParam);
void* ConsumerFunc(void * pParam);
void* SegmentFunc(void * pParam);

void ImageProcess(Mat& );
void GenerateMeanImage(const vector<double>&, Mat& , int , int );
void MergeImage(Mat& , Mat&, Mat& , Mat& );

VideoCapture vpCamera;
VideoCapture vpVideo;
queue<Mat> qmatCameraQueue;
int nWaitKey = -1;
int nFrameCount = 1;
double dPreProcessTime = 0.0f;
double dTotolTime = 0.0f;
Mat matSegmentationImage;
Mat matColorResultMat;
Mat matGrayResultMat;

int main(int argc, char **argv) {
    string strTrainFile = "model/pspnet101_VOC2012.caffemodel";
    string strModelFile = "prototxt/pspnet101_VOC2012_473.prototxt";
    string strVideo = "video/LED091.mp4";
    
    vpCamera.open(0);
    vpVideo.open(strVideo);
    
    if(!vpCamera.isOpened()||!vpVideo.isOpened()){
        cerr << "Camera or video doesn't open!" << endl;
        exit(-1);
    }
    
    long lVideoFN = vpVideo.get(CV_CAP_PROP_FRAME_COUNT);
    long lFrameToStart = 30l;
    vpVideo.set(CV_CAP_PROP_POS_FRAMES, lFrameToStart);
    long lFrameToStop = lVideoFN - 30l;
    if(lFrameToStop < lFrameToStart){
        cout << "Stop frame number is less than start number!" << endl;
        exit(-1);
    }else{
        cout << "The stop frame number is:" << endl;
    }
    double dRate = vpVideo.get(CV_CAP_PROP_FPS);
    cout << "Video FPS is:" << dRate << endl;
    
    int nFrameWidth = 640;
    int nFrameHeight = 480;
    
    vpCamera.set(CV_CAP_PROP_FRAME_WIDTH, nFrameWidth);
    vpCamera.set(CV_CAP_PROP_FRAME_HEIGHT, nFrameHeight); 
    
    
    Mat matMeanColorValue;
    //Mat matSrcMat;
    //int nCropWidth = 2049;
    //int nCropHeight = 1025;
    
    int nCropWidth = 473;
    int nCropHeight = 473;
    
    //int nCropWidth = 1280;
    //int nCropHeight = 720;
    
    GenerateMeanImage(vdMeanColorValue, matMeanColorValue, nCropWidth, nCropHeight);
    
    SegClassifier scVideoSeg(strModelFile,
                             strTrainFile,
                             matMeanColorValue,
                             nCropWidth,
                             nCropHeight
                             /*const string& label_file*/);
    matColorResultMat = Mat(nFrameHeight, nFrameWidth, CV_8UC3, Scalar(0.0,0.0,0.0));
    matGrayResultMat = Mat(nFrameHeight, nFrameWidth, CV_8UC1, Scalar(0.0,0.0,0.0));
    
    int nDevNum = 0;
    namedWindow("CameraCapture");
    int ntrProducter = pthread_create(&pthreadProducter, NULL, ProductFunc, NULL);
    int ntrConsumer = pthread_create(&pthreadConsumer, NULL, ConsumerFunc, NULL);
    int ntrSegmentation = pthread_create(&pthreadSegmentation, NULL, SegmentFunc, &scVideoSeg);
    
    cout << "The Producter thread creating result is:" << ntrProducter << endl;
    cout << "The Consumer thread creating result is:" << ntrProducter << endl;
    if(0 != ntrProducter){
        cerr << "Producter thread doesn't create!" << endl;
        if(ntrConsumer == 0 || ntrSegmentation == 0){
            while(!pthread_cancel(pthreadConsumer)){}
            while(!pthread_cancel(pthreadSegmentation)){}
        }
        else{
            exit(-1);
        }
        pthread_join(pthreadConsumer, NULL);
        pthread_join(pthreadSegmentation, NULL);
        exit(-1);
    }
    
    if(0 != ntrConsumer){
        cerr << "Consumer thread doesn't create!" << endl;
        if(ntrProducter == 0 || ntrSegmentation == 0){
            while(!pthread_cancel(pthreadProducter)){}
            while(!pthread_cancel(pthreadSegmentation)){}
        }
        else{
            exit(-1);
        }
        pthread_join(pthreadProducter, NULL);
        pthread_join(pthreadSegmentation, NULL);
        exit(-1);
    }
    
    if(0 != ntrSegmentation){
        cerr << "Consumer thread doesn't create!" << endl;
        if(ntrProducter == 0 || ntrConsumer == 0){
            while(!pthread_cancel(pthreadProducter)){}
            while(!pthread_cancel(pthreadConsumer)){}
        }
        else{
            exit(-1);
        }
        pthread_join(pthreadProducter, NULL);
        pthread_join(pthreadConsumer, NULL);
        exit(-1);
    }
        
    pthread_join(pthreadProducter, NULL);
    pthread_join(pthreadConsumer, NULL);
    pthread_join(pthreadSegmentation, NULL);

//     while(27 != nWaitKey){
//         vpCamera >> matCameraFrame;
//         cout << "The frame size is:" << matCameraFrame.cols << "*" << matCameraFrame.rows <<endl;
//         imshow("CameraCapture", matCameraFrame);
//         nWaitKey = waitKey(1);
//     }

    return 0;
}


void* ProductFunc(void * pParam){
    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL); 
    pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, NULL); 
    
    while(27 != nWaitKey){
        sleep(0.001);
        pthread_mutex_lock(&ptmMutex);
        Mat matCameraFrame;
        vpCamera >> matCameraFrame;
        qmatCameraQueue.push(matCameraFrame);
        cout << "Producter function run!" << endl;
        cout << qmatCameraQueue.size() << endl;
        pthread_cond_signal(&ptmCond);
        pthread_mutex_unlock(&ptmMutex);
        pthread_testcancel();
    }

    
}

void* ConsumerFunc(void * pParam){
    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL); 
    pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, NULL); 

    //SegClassifier* pscVideoSeg = (SegClassifier*)(pParam);
    while(27 != nWaitKey){
        cout << "Consumer function start!" << endl;
        pthread_mutex_lock(&ptmMutex);
        pthread_mutex_lock(&ptmSegMutex);
        cout << "Consumer function in mutex lock!" << endl;
        while(0 == qmatCameraQueue.size()){
            pthread_cond_wait(&ptmCond, &ptmMutex);
            break;
        }
        
        Mat matCameraFrame = qmatCameraQueue.front();
        matSegmentationImage = matCameraFrame;
        qmatCameraQueue.pop();
        Mat matVideoFrame;
        if(!vpVideo.read(matVideoFrame)){
            cout << "Read video frame error!" << endl;
            exit(-1);
        }
        
        
        double dCurrentImageProcessTime = (double)getTickCount();
        //pscVideoSeg->Classify(matCameraFrame);
        ImageProcess(matCameraFrame);
        
        matVideoFrame = matVideoFrame(Rect(static_cast<int>(0.2 * matVideoFrame.cols), static_cast<int>(matVideoFrame.rows * 0.2f), static_cast<int>(0.6 * matVideoFrame.cols), static_cast<int>(0.6 * matVideoFrame.rows)));
        resize(matVideoFrame, matVideoFrame, Size(matCameraFrame.cols, matCameraFrame.rows));
        pthread_cond_signal(&ptmSegCond);
        
        //Mat matColorResultMat = pscVideoSeg->GetColorResult();

        Mat matPersonFrame = Mat::zeros(matCameraFrame.rows, matCameraFrame.cols, CV_8UC3);
        Mat matPersonVideoFrame = Mat::zeros(matCameraFrame.rows, matCameraFrame.cols, CV_8UC3);
        Mat matthresholdMat;
        matGrayResultMat.convertTo(matthresholdMat, CV_32FC1);
        GaussianBlur(matthresholdMat, matthresholdMat, Size(25, 25), 20.0, 20.0);
        //threshold(matGrayResultMat, matthresholdMat, 100, 255, THRESH_BINARY_INV);
        //matCameraFrame.copyTo(matPersonFrame, matGrayResultMat);
        //matVideoFrame.copyTo(matPersonVideoFrame, matthresholdMat);
        double dImageProcessTime = ((double)getTickCount() - dCurrentImageProcessTime) / getTickFrequency();
        cout << "Image process time is:" << dImageProcessTime <<endl;
        cout << "The image size is " << matCameraFrame.cols << "*" <<  matCameraFrame.rows << endl;

        Mat matDisplayResultMat = Mat(matCameraFrame.rows, matCameraFrame.cols, CV_8UC3);
        MergeImage(matDisplayResultMat, matCameraFrame, matVideoFrame, matthresholdMat);
        //Mat matDisplayResultMat =  matPersonVideoFrame +  matPersonFrame;
        cout << "Result image size is:" << matDisplayResultMat.rows << "," << matDisplayResultMat.cols << endl; 
        double dCurrentProcessTime = (double)(getTickCount());
        double dProcessTime = (dCurrentProcessTime - dPreProcessTime) / getTickFrequency();
        dPreProcessTime = dCurrentProcessTime;
        cout << dProcessTime <<endl;

        cout << "Consumer function run!" << endl;
        dTotolTime += dProcessTime;
        nFrameCount = ++nFrameCount % INT_MAX;
        cout << "nFrameCount:" << nFrameCount << endl;
        cout << "INT_MAX:" << INT_MAX << endl; 
        char cFps[100] ;
        double dfps; 
        if(!(nFrameCount%10)){
            dProcessTime = dTotolTime / 10.0;
            dfps = 1.0 / dProcessTime;
            sprintf(cFps, "FPS:%lf", dfps);
            putText(matDisplayResultMat, cFps, Point(matCameraFrame.cols - 150,30), FONT_HERSHEY_DUPLEX, 0.6, Scalar(0.0, 255.0, 255.0), 2);
            dTotolTime = 0.0;
        }else{
            putText(matDisplayResultMat, cFps, Point(matCameraFrame.cols - 150,30), FONT_HERSHEY_DUPLEX, 0.6, Scalar(0.0, 255.0, 255.0), 2);
        }
        cout << "dTotolTime:" << dTotolTime << endl;
        cout << "FPS:" << dfps << endl;
        imshow("CameraCapture", matDisplayResultMat );
        nWaitKey = waitKey(1);
        pthread_testcancel();
        pthread_mutex_unlock(&ptmSegMutex);
        pthread_mutex_unlock(&ptmMutex);

    }

    cout << "Consumer function end!" << endl;
    
}

void* SegmentFunc(void * pParam){
    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL); 
    pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, NULL); 

    SegClassifier* pscVideoSeg = (SegClassifier*)(pParam);
    while(27 != nWaitKey){
        cout << "Segmentation function start!" << endl;
        pthread_mutex_lock(&ptmSegMutex);
        cout << "Segmentation function in mutex lock!" << endl;
        pthread_cond_wait(&ptmSegCond, &ptmSegMutex);
        
        double dCurrentImageProcessTime = (double)getTickCount();
        pscVideoSeg->Classify(matSegmentationImage);
        
        matGrayResultMat = pscVideoSeg->GetGrayResult();

        pthread_testcancel();
        pthread_mutex_unlock(&ptmSegMutex);

    }

    cout << "Segmentation function end!" << endl;
    
}

void ImageProcess(Mat& matSrcImage){
    if(matSrcImage.empty()){
        cerr << "Input image of ImageProcess function is empty!" << endl;
        exit(-1);
    }
    
    flip(matSrcImage, matSrcImage, 1);
    /*matSrcImage = matSrcImage(Rect((matSrcImage.cols - 640) /2,
                                   (matSrcImage.rows - 720) /2,
                                   640,
                                   720
    ));*/
    
    //medianBlur(matSrcImage, matSrcImage, 3);
    //GaussianBlur(matSrcImage, matSrcImage, Size(3, 3), 0.0, 0.0 );
}

void GenerateMeanImage(const vector<double>&vdMeanColorValue, Mat& matColorMat, int nCropWidth, int nCropHeight){
    if(3 != vdMeanColorValue.size()){
        cerr << "Mean color value error!" << endl;
        exit(-1);
    }
    
    if(matColorMat.empty()){
        matColorMat = Mat(nCropHeight, nCropWidth, CV_32FC3);
    }
       
    Vec3f v3fColorIndex;
    v3fColorIndex[0] = (float)(vdMeanColorValue[0]);
    v3fColorIndex[1] = (float)(vdMeanColorValue[1]);
    v3fColorIndex[2] = (float)(vdMeanColorValue[2]);
    
    int r, c;
    int nTempR;
    Vec3f* pv3fColorValue = matColorMat.ptr<Vec3f>(0);
    for(r = 0; r < nCropHeight; r++){
        nTempR = r * nCropWidth;
        for(c = 0; c < nCropWidth; c++){
            pv3fColorValue[nTempR + c] = v3fColorIndex;
        }
    }    
}

void MergeImage(Mat& matDisplayResultMat, Mat&matCameraFrame, Mat& matVideoFrame, Mat& matthresholdMat){
    if(matDisplayResultMat.empty() || matCameraFrame.empty() || matVideoFrame.empty() || matthresholdMat.empty()){
        cerr << "MergeImage function error!" << endl;
        exit(-1);
    }
    
    if(matDisplayResultMat.size() != matCameraFrame.size() ||
       matDisplayResultMat.size() != matVideoFrame.size() ||
       matDisplayResultMat.size() != matthresholdMat.size() ||
       matCameraFrame.size() != matVideoFrame.size() ||
       matCameraFrame.size() != matthresholdMat.size() ||
       matVideoFrame.size() != matthresholdMat.size()){
        cerr << "MergeImage input image size error!" <<endl;
        exit(-1);
    }
    
    int r, c;
    
    int nmatWidth = matDisplayResultMat.cols;
    int nmatHeight = matDisplayResultMat.rows;
    float* pmatthresholdMat = matthresholdMat.ptr<float>(0);
    Vec3b* pv3bDRMat = matDisplayResultMat.ptr<Vec3b>(0);
    Vec3b* pv3bCFMat = matCameraFrame.ptr<Vec3b>(0);
    Vec3b* pv3bVFMat = matVideoFrame.ptr<Vec3b>(0);
    int nTempR;
    float fRatio;
    for(r = 0; r < matDisplayResultMat.rows; r++){
        nTempR = r * nmatWidth;
        for(c = 0; c < matDisplayResultMat.cols; c++){
            fRatio = pmatthresholdMat[nTempR + c]/ 255.0f;
            pv3bDRMat[nTempR + c][0] = fRatio * pv3bCFMat[nTempR + c][0] + (1.0f - fRatio) * pv3bVFMat[nTempR + c][0];
            pv3bDRMat[nTempR + c][1] = fRatio * pv3bCFMat[nTempR + c][1] + (1.0f - fRatio) * pv3bVFMat[nTempR + c][1];
            pv3bDRMat[nTempR + c][2] = fRatio * pv3bCFMat[nTempR + c][2] + (1.0f - fRatio) * pv3bVFMat[nTempR + c][2];
        }
    }
}
