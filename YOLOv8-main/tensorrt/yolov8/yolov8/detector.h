#pragma once
#include<string>
#include<ie/inference_engine.hpp>
#include<samples/ocv_common.hpp>
#include<ngraph/ngraph.hpp>
#include<opencv2/opencv.hpp>
#include<time.h>
#include <opencv2/opencv.hpp>
#include "samples/args_helper.hpp"
#include "samples/common.hpp"
#include "samples/classification_results.h"
#include<string>


using namespace InferenceEngine;
using namespace std;
using namespace cv;
typedef unsigned char unit8_t;
struct alignas(float) Detection {
	float bbox[4];
	float conf;
	float class_id;
};
class Yolov8Dectector
{
public:
	Yolov8Dectector();
	~Yolov8Dectector();

    //初始化
	bool InitializeDetector(string device, string xml_path);
	//推理
	void process_frame(unsigned char* image_batch, unsigned char* prediction_batch,
		int width, int height, int smallestMax , int batch_size );
	bool uninit();
private:
	float fill_tensor_data_image(ov::Tensor& input_tensor, const cv::Mat& input_image);
	void convert(const cv::Mat& input, cv::Mat& output, const bool normalize, const bool exchangeRB);
	void readClassFile(const std::string& class_file, std::map<int, std::string>& labels);

private:
	ov::Core core;
	std::shared_ptr<ov::Model> model;
	ov::CompiledModel compiled_model;
	string _inputname;
	string _onnx_path;
	string _device;
	int _batch_size;
	//模型输入图片大小
	int _model_input_w;
	int _model_input_h;
	int _model_input_c;
	int _model_inputSize;
	int nums = 0;
	float kNmsThresh;
	float kConfThresh ;
	int _model_output_w;
	int _model_output_h;
	int _model_output_class;
	std::string class_file = "D:\\AI\\TRT\\yolov8_tensorrt-main\\weights\\classes7.txt";
	//原图片大小
	int _org_h;
	int _org_w;

	int _buffer_size;
	const std::vector<std::string> class_names = { "rubber stopper", "push rod tail", "needle tail", "mouth", "crooked mouth", "screw mouth", "small rubber plug" };
};
