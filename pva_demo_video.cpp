#include <stdio.h>  // for snprintf
#include <string>
#include <vector>
#include <math.h>
#include <fstream>
#include <boost/python.hpp>
#include "caffe/caffe.hpp"
#include "gpu_nms.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace caffe;
using namespace std;

#define max(a, b) (((a)>(b)) ? (a) :(b))
#define min(a, b) (((a)<(b)) ? (a) :(b))
const int class_num= 21;

/*
 * ===  Class  ======================================================================
 *         Name:  Detector
 *  Description:  FasterRCNN CXX Detector
 * =====================================================================================
 */
class Detector {
public:
    Detector(const string& model_file, const string& weights_file);
    void Detection(const string& im_name);
    void bbox_transform_inv(const int num, const float* box_deltas, const float* pred_cls, float* boxes, float* pred, int img_height, int img_width);
    void vis_detections(cv::Mat image, int* keep, int num_out, float* sorted_pred_cls, float CONF_THRESH);
    void boxes_sort(int num, const float* pred, float* sorted_pred);
private:
    shared_ptr<Net<float> > net_;
    Detector(){}
};

/*
 * ===  FUNCTION  ======================================================================
 *         Name:  Detector
 *  Description:  Load the model file and weights file
 * =====================================================================================
 */
//load modelfile and weights
Detector::Detector(const string& model_file, const string& weights_file)
{
    net_ = shared_ptr<Net<float> >(new Net<float>(model_file, caffe::TEST));
    net_->CopyTrainedLayersFrom(weights_file);
}


//Using for box sort
struct Info
{
    float score;
    const float* head;
};
bool compare(const Info& Info1, const Info& Info2)
{
    return Info1.score > Info2.score;
}

/*
 * ===  FUNCTION  ======================================================================
 *         Name:  Detect
 *  Description:  Perform detection operation
 *                 Warning the max input size should less than 1000*600
 * =====================================================================================
 */
//perform detection operation
//input image max size 1000*600
void Detector::Detection(const string& im_name)
{
	// configure parameters in /pva-faster-rcnn/models/pvanet/cfgs/submit_1019.yml
    float CONF_THRESH = 0.8;	// confidence thresh
    float NMS_THRESH = 0.4;
    const int SCALE_MULTIPLE_OF = 32;
    const int MAX_SIZE = 2000;
    const int SCALES = 640;
    
    //const int  max_input_side=2000;
    //const int  min_input_side=640;

    cv::Mat cv_img = cv::imread(im_name);
    cv::Mat cv_new(cv_img.rows, cv_img.cols, CV_32FC3, cv::Scalar(0,0,0));
    if(cv_img.empty())
    {
        std::cout<<"Can not get the image file !"<<endl;
        return ;
    }
    
    //int max_side = max(cv_img.rows, cv_img.cols);
    //int min_side = min(cv_img.rows, cv_img.cols);

    //float max_side_scale = float(max_side) / float(max_input_side);
    //float min_side_scale = float(min_side) /float( min_input_side);
    //float max_scale=max(max_side_scale, min_side_scale);

    //float img_scale = 1;

	/*
    if(max_scale > 1)
    {
        img_scale = float(1) / max_scale;
    }*/
	
	int im_size_min = min(cv_img.rows, cv_img.cols);
	int im_size_max = max(cv_img.rows, cv_img.cols);
	float im_scale = float(SCALES) / im_size_min;// if im_scale have many value, it will support muti-scale detection
	if (round(im_scale * im_size_max) > MAX_SIZE)
		im_scale = float(MAX_SIZE) / im_size_max;
	// Make width and height be multiple of a specified number
	float im_scale_x = std::floor(cv_img.cols * im_scale / SCALE_MULTIPLE_OF) * SCALE_MULTIPLE_OF / cv_img.cols;
	float im_scale_y = std::floor(cv_img.rows * im_scale / SCALE_MULTIPLE_OF) * SCALE_MULTIPLE_OF / cv_img.rows;
    // keep image size less than 2000 * 640
    int height = int(cv_img.rows * im_scale_y);
    int width = int(cv_img.cols * im_scale_x);
    int num_out;
    cv::Mat cv_resized;

    std::cout<<"imagename "<<im_name<<endl;
    float im_info[6];	// (cv_resized)image's height, width, scale(equal 1 or 1/max_scale)
    float data_buf[height*width*3];	// each pixel value in cv_resized
    float *boxes = NULL;
    float *pred = NULL;
    float *pred_per_class = NULL;
    float *sorted_pred_cls = NULL;
    int *keep = NULL;
    const float* bbox_delt;
    const float* rois;
    const float* pred_cls;
    int num;

    for (int h = 0; h < cv_img.rows; ++h )
    {
        for (int w = 0; w < cv_img.cols; ++w)
        {
	    // mins 102.9801, 115.9465, 122.7717, why do it? 
	    // we can instead of below by cv_new = cv.img - cv::Mat(img.cols, img.rows, CV_32FC3, cv::Scalar(102.9801, 115.9465, 122.7717))
            cv_new.at<cv::Vec3f>(cv::Point(w, h))[0] = float(cv_img.at<cv::Vec3b>(cv::Point(w, h))[0])-float(102.9801);
            cv_new.at<cv::Vec3f>(cv::Point(w, h))[1] = float(cv_img.at<cv::Vec3b>(cv::Point(w, h))[1])-float(115.9465);
            cv_new.at<cv::Vec3f>(cv::Point(w, h))[2] = float(cv_img.at<cv::Vec3b>(cv::Point(w, h))[2])-float(122.7717);

        }
    }

    cv::resize(cv_new, cv_resized, cv::Size(width, height));
    im_info[0] = cv_resized.rows;
    im_info[1] = cv_resized.cols;
    im_info[2] = im_scale_x;
    im_info[3] = im_scale_y;
	im_info[4] = im_scale_x;
	im_info[5] = im_scale_y;
	// sequence data
    for (int h = 0; h < height; ++h )
    {
        for (int w = 0; w < width; ++w)
        {
            data_buf[(0*height+h)*width+w] = float(cv_resized.at<cv::Vec3f>(cv::Point(w, h))[0]);
            data_buf[(1*height+h)*width+w] = float(cv_resized.at<cv::Vec3f>(cv::Point(w, h))[1]);
            data_buf[(2*height+h)*width+w] = float(cv_resized.at<cv::Vec3f>(cv::Point(w, h))[2]);
        }
    }

    net_->blob_by_name("data")->Reshape(1, 3, height, width);
    net_->blob_by_name("data")->set_cpu_data(data_buf);
    net_->blob_by_name("im_info")->set_cpu_data(im_info);
    net_->ForwardFrom(0);
    bbox_delt = net_->blob_by_name("bbox_pred")->cpu_data();	// bbox_delt is offset ratio of bounding box, get by bounding box regression
    num = net_->blob_by_name("rois")->num();	// number of region proposals


    rois = net_->blob_by_name("rois")->cpu_data();	// scores and bounding boxes coordinate
    pred_cls = net_->blob_by_name("cls_prob")->cpu_data();
    boxes = new float[num*4];
    pred = new float[num*5*class_num];
    pred_per_class = new float[num*5];
    sorted_pred_cls = new float[num*5];
    keep = new int[num];	// index of bounding box?

    for (int n = 0; n < num; n++)
    {
        for (int c = 0; c < 4; c++)
        {
        	// resize function may increase img size, if that, we should increase bounding boxes coordinate
            boxes[n*4+c] = rois[n*5+c+1] / im_info[c + 2];
        }
    }

    bbox_transform_inv(num, bbox_delt, pred_cls, boxes, pred, cv_img.rows, cv_img.cols);
    for (int i = 1; i < class_num; i ++)		// i = 0, means background
    {
        for (int j = 0; j< num; j++)
        {
            for (int k=0; k<5; k++)
                pred_per_class[j*5+k] = pred[(i*num+j)*5+k];
        }
        boxes_sort(num, pred_per_class, sorted_pred_cls);
        // you can read vis_detection() function to understand these variables
        // @num_out: after nms, the number of bounding box
        // @keep: after nms, the index of bounding box in sorted_pred_cls
        _nms(keep, &num_out, sorted_pred_cls, num, 5, NMS_THRESH, 0);
        vis_detections(cv_img, keep, num_out, sorted_pred_cls, CONF_THRESH);
    }

    cv::imwrite("vis.jpg",cv_img);
    delete []boxes;
    delete []pred;
    delete []pred_per_class;
    delete []keep;
    delete []sorted_pred_cls;

}

/*
 * ===  FUNCTION  ======================================================================
 *         Name:  vis_detections
 *  Description:  Visuallize the detection result
 * =====================================================================================
 */
void Detector::vis_detections(cv::Mat image, int* keep, int num_out, float* sorted_pred_cls, float CONF_THRESH)
{
    int i=0;
    while(sorted_pred_cls[keep[i]*5+4]>CONF_THRESH && i < num_out)
    {
        if(i>=num_out)
            return;
        cv::rectangle(image,cv::Point(sorted_pred_cls[keep[i]*5+0], sorted_pred_cls[keep[i]*5+1]),cv::Point(sorted_pred_cls[keep[i]*5+2], sorted_pred_cls[keep[i]*5+3]),cv::Scalar(255,0,0));
        i++;
    }
}

/*
 * ===  FUNCTION  ======================================================================
 *         Name:  boxes_sort
 *  Description:  Sort the bounding box according score
 * =====================================================================================
 */
void Detector::boxes_sort(const int num, const float* pred, float* sorted_pred)
{
    vector<Info> my;
    Info tmp;
    for (int i = 0; i< num; i++)
    {
        tmp.score = pred[i*5 + 4];
        tmp.head = pred + i*5;
        my.push_back(tmp);
    }
    std::sort(my.begin(), my.end(), compare);
    for (int i=0; i<num; i++)
    {
        for (int j=0; j<5; j++)
            sorted_pred[i*5+j] = my[i].head[j];	// sequence data
    }
}

/*
 * ===  FUNCTION  ======================================================================
 *         Name:  bbox_transform_inv
 *  Description:  Compute bounding box regression value
 * =====================================================================================
 */
void Detector::bbox_transform_inv(int num, const float* box_deltas, const float* pred_cls, float* boxes, float* pred, int img_height, int img_width)
{
    float width, height, ctr_x, ctr_y, dx, dy, dw, dh, pred_ctr_x, pred_ctr_y, pred_w, pred_h;
    for(int i=0; i< num; i++)
    {
        width = boxes[i*4+2] - boxes[i*4+0] + 1.0;
        height = boxes[i*4+3] - boxes[i*4+1] + 1.0;
        ctr_x = boxes[i*4+0] + 0.5 * width;		// (ctr_x, ctr_y) is center coordinater of bounding box 
        ctr_y = boxes[i*4+1] + 0.5 * height;
        for (int j=0; j< class_num; j++)
        {

            dx = box_deltas[(i*class_num+j)*4+0];
            dy = box_deltas[(i*class_num+j)*4+1];
            dw = box_deltas[(i*class_num+j)*4+2];
            dh = box_deltas[(i*class_num+j)*4+3];
            pred_ctr_x = ctr_x + width*dx;
            pred_ctr_y = ctr_y + height*dy;
            pred_w = width * exp(dw);
            pred_h = height * exp(dh);
            pred[(j*num+i)*5+0] = max(min(pred_ctr_x - 0.5* pred_w, img_width -1), 0);	// avoid over boundary
            pred[(j*num+i)*5+1] = max(min(pred_ctr_y - 0.5* pred_h, img_height -1), 0);
            pred[(j*num+i)*5+2] = max(min(pred_ctr_x + 0.5* pred_w, img_width -1), 0);
            pred[(j*num+i)*5+3] = max(min(pred_ctr_y + 0.5* pred_h, img_height -1), 0);
            pred[(j*num+i)*5+4] = pred_cls[i*class_num+j];
        }
    }

}

int main()
{
    string model_file = "/home/mordekaiser/pva-faster-rcnn/models/pvanet/pva9.1/faster_rcnn_train_test_21cls.pt";
    string weights_file = "/home/mordekaiser/pva-faster-rcnn/models/pvanet/pva9.1/PVA9.1_ImgNet_COCO_VOC0712.caffemodel";
    //int GPUID=0;
    //Caffe::SetDevice(GPUID);
    Caffe::set_mode(Caffe::CPU);
    Detector det = Detector(model_file, weights_file);
    //det.Detection("/home/mordekaiser/pva-demo/pic/004545.jpg");
    cv::VideoCapture capture("/home/mordekaiser/pva-video/YUN00004.mp4");
    if (!capture.isOpened()) {
    	std::cout << "No input video" << endl;
    	return 1;
    }
    
    double fps = capture.get(cv::CV_CAP_PROP_FPS);
    bool stop(false);
    
    cv::Mat frame;
    cv::namedWindow("Extracted Frame");
    
    int delay = 1000 / fps;
    while(!stop) {
    	if (!capture.read(frame))
    		break;
    	cv::imshow("Extracted Frame", frame);
    	if (cv::waitKey(delay) >= 0)
    	stop = true;
    }
    
    return 0;
    
return 0;
}
