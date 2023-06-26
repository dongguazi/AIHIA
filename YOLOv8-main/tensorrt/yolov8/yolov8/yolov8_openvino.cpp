
#include"detector.h";

using namespace InferenceEngine;
using namespace std;
using namespace cv;
typedef unsigned char unit8_t;

std::string DEVICE = "CPU";

//onnx fp16	D:\\AI\\YOLO\\yolov8-main\\runs\\detect\\train3\\weights\\best.onnx
// onnx fp32 D:\\AI\\YOLO\\yolov8-data\\YOLOv8-main\\runs\\detect\\train\\weights\\best.onnx
//  D:\\IROnnx\\best_fp16.onnx

std::string IR_filename_ONNX = "D:\\IROnnx\\best_fp32.onnx";

//D:\\AI\\YOLO\\yolov8-main\\runs\\detect\\train3\\weights\\IR_FP16\\best.xml
//D:\\AI\\YOLO\\yolov8-data\\YOLOv8-main\\runs\\detect\\train\\weights\\best.xml
std::string IR_filename_IR = "D:\\AI\YOLO\\yolov8-main\\runs\\detect\\train3\\weights\\best_openvino_model\\best.xml";
string images_path = "E:\\DataSets\\keypoint\\images\\testImages";

//��Ҫ����ʵ��������õĲ���
int buff_size = 9999;
int buff_size_Out = 99999999;

int width = 3072;
int height = 2048;

int batch_size = 1;

int cycle = 1;

int main()
{
	//1.��������
	Yolov8Dectector* detector = new Yolov8Dectector();

	//2.��ʼ��ģ��
	detector->InitializeDetector(DEVICE, IR_filename_IR);

	//3.ͼ��ǰ����
	vector<cv::String> filenames;
	glob(images_path, filenames, false);
	int images_count = filenames.size();

	vector<Mat> images_all_list;
	vector<Mat> images_batch_list;

	//4.�����ж�ȡ��ͼƬ����
	for (int i = 0; i < images_count; i++)
	{
		cv::Mat img = imread(filenames[i]);
		//width = images_batch_list[i].cols;
		//height = images_batch_list[i].rows;
		// 
		images_all_list.push_back(img);
	}

	int epoch = images_all_list.size() / batch_size;

	//5.����batch��С�ָ��epoch
	for (size_t i = 0; i < epoch; i++)
	{
		for (size_t j = 0; j < batch_size; j++)
		{
			Mat img = images_all_list[i * batch_size + j].clone();
			images_batch_list.push_back(img);
		}
	}

	buff_size_Out = batch_size * height * width * 6;

	cout << "*****4.��ʼ����" << endl;

	//6.�������У�����Ԥ��������н��
	  //ѭ��ÿ��epoch��ÿ�ζ����Ԥ��Ľ��
	for (size_t c = 0; c < cycle; c++)
	{
		cout << "~~~~~~~~~~~~~~~~~~�ڴ�й©ѭ��������" << c << "~~~~~~~~~~~~~~~~~~~~~~" << endl;
		for (size_t epoch_i = 0; epoch_i < epoch; epoch_i++)
		{
			cout << endl;
			cout << "   +++++��ʼ��" << epoch_i << "���ڣ�" << endl;

			auto run_begintime = cv::getTickCount();

			int per_pic_size = width * height * 3;

			unsigned char* img_batch = new unsigned char[per_pic_size * batch_size];

			//��batch��ÿ��ͼƬ���뵽ͼƬָ��0,1,2
			for (size_t batch_size_i = 0; batch_size_i < batch_size; batch_size_i++)
			{
				//cv::imwrite("org.jpg", images_batch_list[epoch_i * batch_size + batch_size_i]);
				//������batch��ͼƬ���ص�ͼƬָ����
				memcpy(img_batch + batch_size_i * per_pic_size, images_batch_list[epoch_i * batch_size + batch_size_i].data, per_pic_size * sizeof(unsigned char));
			}

			unsigned char* prediction_batch = new unsigned char[buff_size_Out];

			//������batch��ÿ��ͼƬ���������Ԥ��
			//void process_frame(unsigned char* image_batch, unsigned char* prediction_batch,
			//	int width, int height, int smallestMax = 512, int batch_size = 1);
			detector->process_frame(img_batch, prediction_batch, width, height, 512, batch_size);

			int num = 0;
			auto run_endtime = cv::getTickCount();
			auto infer_time = (to_string)((run_endtime - run_begintime) * 1000 / getTickFrequency() / batch_size);
			cout << "  +++++��" << epoch_i << "���ڣ�����" << batch_size << "��ͼƬ������ͼƬ��ƽ��ʱ��ms:" << infer_time << endl;

			delete[] img_batch;
			delete[] prediction_batch;
		}
		cout << "      �����ɣ�" << endl;
	}

	delete detector;
	waitKey(0);
	//destroyAllWindows();
	return 0;
}