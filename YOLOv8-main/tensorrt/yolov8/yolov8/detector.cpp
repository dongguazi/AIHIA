#include"detector.h";

Yolov8Dectector::Yolov8Dectector() {};
Yolov8Dectector::~Yolov8Dectector() {};

std::vector<string> outNames;
/// <summary>
/// 初始化网络
/// </summary>
/// <param name="device">设备="CPU""GPU"</param>
/// <param name="onnx_path">xml或者onnx文件路径</param>
bool Yolov8Dectector::InitializeDetector(string device, string onnx_path)
{
	_onnx_path = onnx_path;
	_device = device;

	// -------- Step 1. Initialize OpenVINO Runtime Core --------
	//ov::Core core;
	auto  devices = core.get_available_devices();
	for (auto iter = devices.begin(); iter != devices.end(); iter++)
	{
		cout << "当前可用设备名称：" << iter->c_str() << endl;
	}

	// -------- Step 2. Read a model --------
	try
	{
		auto load_begintime = cv::getTickCount();

		model = core.read_model(_onnx_path);

		//printInputAndOutputsInfo(*model);
		//cout << "input_names_size:" << model->input().get_names().size() << endl;
		//cout << "input_tensor:" << model->input().get_tensor() << endl;
		//cout << "input_name:" << model->input().get_any_name() << endl;
		//OPENVINO_ASSERT(model->outputs().size() == 4, "Sample supports models with 1 output only");
		
		cout << "input_type:" << model->input().get_element_type() << endl;
		cout << "input_shape:" << model->input().get_shape() << endl;

		cout << "outputNums:" <<model->outputs().size() << endl;
		auto outputs = model->outputs();

		for (size_t i = 0; i < model->outputs().size(); i++)
		{
			cout << "output_names:" << i << ":" << model->outputs()[i].get_any_name() << endl;
			cout << "output_type:"<<i<<":" << model->get_output_element_type(i) << endl;
			cout << "output_shape:"<<i << ":" << model->get_output_shape(i) << endl;
			outNames.push_back(model->outputs()[i].get_any_name());
		}

		auto load_endtime = cv::getTickCount();
		auto infer_time = (to_string)((load_endtime - load_begintime) * 1000 / getTickFrequency());
		cout << "*模型加载成功！" << device << "  *加载时间ms:" << infer_time << endl;
	}
	catch (const std::exception& ex)
	{
		cout << "*模型加载失败！" << device << endl;
		return false;
	}

	// -------- Step 3. Configure preprocessing --------
	ov::preprocess::PrePostProcessor ppp(model);
	// 1) input() with no args assumes a model has a single input
	ov::preprocess::InputInfo& input_info = ppp.input();
	// 2) Set input tensor information:
	// - precision of tensor is supposed to be 'u8',some model need f32 or f16
	// - layout of data is 'NHWC'
	const ov::Layout tensor_layout{ "NCHW" };
	input_info.tensor().set_element_type(ov::element::f32).set_layout(tensor_layout);
	input_info.preprocess().resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
	//input_info.preprocess().convert_color(ov::preprocess::ColorFormat::RGB);

	// 3) Here we suppose model has 'NCHW' layout for input
	input_info.model().set_layout("NCHW");

	// 4) output() with no args assumes a model has a single result
	// - precision of tensor is supposed to be 'f32'
	ppp.output().tensor().set_element_type(ov::element::f32);

	// 5) Once the build() method is called, the pre(post)processing steps
	// for layout and precision conversions are inserted automatically
	model = ppp.build();
	
	//设置模型的batch size
	ov::set_batch(model, 1);

	// -------- Step 4. Loading model to the device --------
	auto load_begintime = cv::getTickCount();
	compiled_model = core.compile_model(model, _device);
	auto load_endtime = cv::getTickCount();
	auto load_time = (to_string)((load_endtime - load_begintime) * 1000 / getTickFrequency());
	cout << "*模型编译时间ms:" << load_time << endl;

	return true;
}


//推理过程
void Yolov8Dectector::process_frame(unsigned char* image_batch, unsigned char* prediction_batch,
	int width, int height, int smallestMax = 512, int batch_size = 1)
{
	//获取原始输入参数
	_org_h = height;
	_org_w = width;
	_batch_size = batch_size;

	// -------- Step 5. Create infer request --------
	ov::InferRequest infer_request = compiled_model.create_infer_request();

	// -------- Step 6. Combine multiple input images as batch --------
	ov::Tensor input_tensor = infer_request.get_input_tensor();

	const size_t image_size = shape_size(model->input().get_shape());
	
	_model_input_h = input_tensor.get_shape()[2];
	_model_input_w = input_tensor.get_shape()[3];

	std::map<int, std::string> labels;
	readClassFile(class_file, labels);

	float* blob_data = input_tensor.data<float>();
	
	vector<Mat> batch_Pic;

	//循环所有图片，进行推理预测
	for (int i = 0; i < batch_size; i++)
	{
		nums ++;
		cv::Mat sourceImage = cv::Mat(_org_h, _org_w, CV_8UC3, image_batch + i * _org_h * _org_w * 3);
		cv::Mat  disImage;
		sourceImage.copyTo(disImage);
		//std::string savePath1 = "org0_" + std::to_string(i) + ".jpg";
		//std::string savePath2 = "org1_" + std::to_string(i) + ".jpg";
		//cv::imwrite(savePath1, sourceImage);
		//cv::imwrite(savePath2, disImage);

		const float factor = fill_tensor_data_image(input_tensor, sourceImage);

		// -------- Step 9. Do asynchronous inference --------
		auto infer_begintime = cv::getTickCount();
		infer_request.set_input_tensor(input_tensor);
		//infer
		infer_request.infer();

		auto infer_endtime = cv::getTickCount();
		auto infer_time = (to_string)((infer_endtime - infer_begintime) * 1000 / getTickFrequency());
		cout << "openvino推理时间ms:" << infer_time << endl;

		// -------- Step 10.  dispose output --------
		const ov::Tensor output = infer_request.get_output_tensor();
		const ov::Shape output_shape = output.get_shape();
	    const float* output_buffer = output.data<float>();

		// 解析推理结果
		const int out_rows = output_shape[1]; //获得"output"节点的rows
		const int out_cols = output_shape[2]; //获得"output"节点的cols
		const cv::Mat det_output(out_rows, out_cols, CV_32F, (float*)output_buffer);

		std::vector<cv::Rect> boxes;
		std::vector<int> class_ids;
		std::vector<float> confidences;
		kNmsThresh = 0.3f;
		kConfThresh = 0.2f;
		// 输出格式是[11,8400], 每列代表一个框(即最多有8400个框), 前面4行分别是cx, cy, ow, oh, 后面80行是每个类别的置信度
		std::cout << std::endl << std::endl;
		for (int i = 0; i < det_output.cols; ++i) {
			const cv::Mat classes_scores = det_output.col(i).rowRange(4, 11);
			cv::Point class_id_point;
			double score;
			cv::minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

			// 置信度 0～1之间
			if (score > 0.2) {
				const float cx = det_output.at<float>(0, i);
				const float cy = det_output.at<float>(1, i);
				const float ow = det_output.at<float>(2, i);
				const float oh = det_output.at<float>(3, i);
				cv::Rect box;
				box.x = static_cast<int>((cx - 0.5 * ow) * factor);
				box.y = static_cast<int>((cy - 0.5 * oh) * factor);
				box.width = static_cast<int>(ow * factor);
				box.height = static_cast<int>(oh * factor);

				boxes.push_back(box);
				class_ids.push_back(class_id_point.y);
				confidences.push_back(score);
			}
		}

		// NMS, 消除具有较低置信度的冗余重叠框
		std::vector<int> indexes;
		cv::dnn::NMSBoxes(boxes, confidences, kConfThresh, kNmsThresh, indexes);
		for (size_t i = 0; i < indexes.size(); i++) {
			const int index = indexes[i];
			const int idx = class_ids[index];
			cv::rectangle(disImage, boxes[index], cv::Scalar(0, 0, 255), 2, 8);
			cv::rectangle(disImage, cv::Point(boxes[index].tl().x, boxes[index].tl().y - 20),
				cv::Point(boxes[index].br().x, boxes[index].tl().y), cv::Scalar(0, 255, 255), -1);
			string nameScore = class_names[idx] + "  " + std::to_string( confidences[idx]);
			cv::putText(disImage, nameScore, cv::Point(boxes[index].tl().x, boxes[index].tl().y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
		}

		std::string savePath = "vino_res/result" + std::to_string(nums) + ".jpg";
		cv::imwrite(savePath, disImage);
	}

}

//图像前处理
float Yolov8Dectector::fill_tensor_data_image(ov::Tensor& input_tensor, const cv::Mat& input_image)
{
	/// letterbox变换: 不改变宽高比(aspect ratio), 将input_image缩放并放置到blob_image左上角
	const ov::Shape tensor_shape = input_tensor.get_shape();
	const size_t num_channels = tensor_shape[1];
	const size_t height = tensor_shape[2];
	const size_t width = tensor_shape[3];
	// 缩放因子
	const float scale = std::min(height / float(input_image.rows),
		width / float(input_image.cols));
	const cv::Matx23f matrix{
		scale, 0.0, 0.0,
		0.0, scale, 0.0,
	};
	cv::Mat blob_image;
	// 下面根据scale范围进行数据转换, 这只是为了提高一点速度(主要是提高了交换通道的速度), 但可读性很差
	// 如果不在意这点速度提升的可以固定一种做法(前两个if分支都可以)
	if (scale > 1.0 + FLT_EPSILON) {
		// 要放大, 那么先交换通道再放大
		convert(input_image, blob_image, true, true);
		cv::warpAffine(blob_image, blob_image, matrix, cv::Size(width, height));
	}
	else if (scale < 1.0 - FLT_EPSILON) {
		// 要缩小, 那么先缩小再交换通道
		cv::warpAffine(input_image, blob_image, matrix, cv::Size(width, height));
		convert(blob_image, blob_image, true, true);
	}
	else {
		convert(input_image, blob_image, true, true);
	}
	//    cv::imshow("input_image", input_image);
	//    cv::imshow("blob_image", blob_image);
	//    cv::waitKey(0);

		/// 将图像数据填入input_tensor
	float* const input_tensor_data = input_tensor.data<float>();
	// 原有图片数据为 HWC格式，模型输入节点要求的为 CHW 格式
	for (size_t c = 0; c < num_channels; c++) {
		for (size_t h = 0; h < height; h++) {
			for (size_t w = 0; w < width; w++) {
				input_tensor_data[c * width * height + h * width + w] = blob_image.at<cv::Vec<float, 3>>(h, w)[c];
			}
		}
	}
	return 1 / scale;
}
/// 转换图像数据: 先转换元素类型, (可选)然后归一化到[0, 1], (可选)然后交换RB通道
void Yolov8Dectector::convert(const cv::Mat& input, cv::Mat& output, const bool normalize, const bool exchangeRB)
{
	input.convertTo(output, CV_32F);
	if (normalize) {
		output = output / 255.0; // 归一化到[0, 1]
	}
	if (exchangeRB) {
		cv::cvtColor(output, output, cv::COLOR_BGR2RGB);
	}
}

void Yolov8Dectector::readClassFile(const std::string& class_file, std::map<int, std::string>& labels) {
	std::fstream file(class_file, std::ios::in);
	if (!file.is_open()) {
		std::cout << "Load classes file failed: " << class_file << std::endl;
		system("pause");
		exit(0);
	}
	std::cout << "Load classes file success: " << class_file << std::endl;
	std::string str_line;
	int index = 0;
	while (getline(file, str_line)) {
		labels.insert({ index, str_line });
		index++;
	}
	file.close();
}



bool Yolov8Dectector::uninit() {
	return true;
}




