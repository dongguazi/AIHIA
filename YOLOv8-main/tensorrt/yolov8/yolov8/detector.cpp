#include"detector.h";

Yolov8Dectector::Yolov8Dectector() {};
Yolov8Dectector::~Yolov8Dectector() {};

std::vector<string> outNames;
/// <summary>
/// ��ʼ������
/// </summary>
/// <param name="device">�豸="CPU""GPU"</param>
/// <param name="onnx_path">xml����onnx�ļ�·��</param>
bool Yolov8Dectector::InitializeDetector(string device, string onnx_path)
{
	_onnx_path = onnx_path;
	_device = device;

	// -------- Step 1. Initialize OpenVINO Runtime Core --------
	//ov::Core core;
	auto  devices = core.get_available_devices();
	for (auto iter = devices.begin(); iter != devices.end(); iter++)
	{
		cout << "��ǰ�����豸���ƣ�" << iter->c_str() << endl;
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
		cout << "*ģ�ͼ��سɹ���" << device << "  *����ʱ��ms:" << infer_time << endl;
	}
	catch (const std::exception& ex)
	{
		cout << "*ģ�ͼ���ʧ�ܣ�" << device << endl;
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
	
	//����ģ�͵�batch size
	ov::set_batch(model, 1);

	// -------- Step 4. Loading model to the device --------
	auto load_begintime = cv::getTickCount();
	compiled_model = core.compile_model(model, _device);
	auto load_endtime = cv::getTickCount();
	auto load_time = (to_string)((load_endtime - load_begintime) * 1000 / getTickFrequency());
	cout << "*ģ�ͱ���ʱ��ms:" << load_time << endl;

	return true;
}


//�������
void Yolov8Dectector::process_frame(unsigned char* image_batch, unsigned char* prediction_batch,
	int width, int height, int smallestMax = 512, int batch_size = 1)
{
	//��ȡԭʼ�������
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

	//ѭ������ͼƬ����������Ԥ��
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
		cout << "openvino����ʱ��ms:" << infer_time << endl;

		// -------- Step 10.  dispose output --------
		const ov::Tensor output = infer_request.get_output_tensor();
		const ov::Shape output_shape = output.get_shape();
	    const float* output_buffer = output.data<float>();

		// ����������
		const int out_rows = output_shape[1]; //���"output"�ڵ��rows
		const int out_cols = output_shape[2]; //���"output"�ڵ��cols
		const cv::Mat det_output(out_rows, out_cols, CV_32F, (float*)output_buffer);

		std::vector<cv::Rect> boxes;
		std::vector<int> class_ids;
		std::vector<float> confidences;
		kNmsThresh = 0.3f;
		kConfThresh = 0.2f;
		// �����ʽ��[11,8400], ÿ�д���һ����(�������8400����), ǰ��4�зֱ���cx, cy, ow, oh, ����80����ÿ���������Ŷ�
		std::cout << std::endl << std::endl;
		for (int i = 0; i < det_output.cols; ++i) {
			const cv::Mat classes_scores = det_output.col(i).rowRange(4, 11);
			cv::Point class_id_point;
			double score;
			cv::minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

			// ���Ŷ� 0��1֮��
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

		// NMS, �������нϵ����Ŷȵ������ص���
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

//ͼ��ǰ����
float Yolov8Dectector::fill_tensor_data_image(ov::Tensor& input_tensor, const cv::Mat& input_image)
{
	/// letterbox�任: ���ı��߱�(aspect ratio), ��input_image���Ų����õ�blob_image���Ͻ�
	const ov::Shape tensor_shape = input_tensor.get_shape();
	const size_t num_channels = tensor_shape[1];
	const size_t height = tensor_shape[2];
	const size_t width = tensor_shape[3];
	// ��������
	const float scale = std::min(height / float(input_image.rows),
		width / float(input_image.cols));
	const cv::Matx23f matrix{
		scale, 0.0, 0.0,
		0.0, scale, 0.0,
	};
	cv::Mat blob_image;
	// �������scale��Χ��������ת��, ��ֻ��Ϊ�����һ���ٶ�(��Ҫ������˽���ͨ�����ٶ�), ���ɶ��Ժܲ�
	// �������������ٶ������Ŀ��Թ̶�һ������(ǰ����if��֧������)
	if (scale > 1.0 + FLT_EPSILON) {
		// Ҫ�Ŵ�, ��ô�Ƚ���ͨ���ٷŴ�
		convert(input_image, blob_image, true, true);
		cv::warpAffine(blob_image, blob_image, matrix, cv::Size(width, height));
	}
	else if (scale < 1.0 - FLT_EPSILON) {
		// Ҫ��С, ��ô����С�ٽ���ͨ��
		cv::warpAffine(input_image, blob_image, matrix, cv::Size(width, height));
		convert(blob_image, blob_image, true, true);
	}
	else {
		convert(input_image, blob_image, true, true);
	}
	//    cv::imshow("input_image", input_image);
	//    cv::imshow("blob_image", blob_image);
	//    cv::waitKey(0);

		/// ��ͼ����������input_tensor
	float* const input_tensor_data = input_tensor.data<float>();
	// ԭ��ͼƬ����Ϊ HWC��ʽ��ģ������ڵ�Ҫ���Ϊ CHW ��ʽ
	for (size_t c = 0; c < num_channels; c++) {
		for (size_t h = 0; h < height; h++) {
			for (size_t w = 0; w < width; w++) {
				input_tensor_data[c * width * height + h * width + w] = blob_image.at<cv::Vec<float, 3>>(h, w)[c];
			}
		}
	}
	return 1 / scale;
}
/// ת��ͼ������: ��ת��Ԫ������, (��ѡ)Ȼ���һ����[0, 1], (��ѡ)Ȼ�󽻻�RBͨ��
void Yolov8Dectector::convert(const cv::Mat& input, cv::Mat& output, const bool normalize, const bool exchangeRB)
{
	input.convertTo(output, CV_32F);
	if (normalize) {
		output = output / 255.0; // ��һ����[0, 1]
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




