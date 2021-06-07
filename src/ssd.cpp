#include "ssd.hpp"

vector<string> VOC_CLASSES = {"__background__", "handrail", "circle", "vertical"};
// vector<string> VOC_CLASSES = {"__background__", "up", "down", "open", "close",
// 								"1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
// 								"11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
// 								"21", "22", "23", "24", "25", "26", "27", "28", "29", "30",
// 								"31", "32", "33"};

SSDetection::SSDetection(const std::string& weight)
{
    SetFixedParams();                   // 设置基本参数
    SetDevice();                        // 设置 Device 类型
    LoadTracedModule(weight);           // 加载模型权重和设备
}

void SSDetection::SetFixedParams()
{
    // SSD 论文中的参数
    net_size_ = 300;
    feature_maps_ = {38, 19, 10, 5, 3, 1};
    steps_ = {8, 16, 32, 64, 100, 300};
    min_size_ = {30, 60, 111, 162, 213, 264};
    max_size_ = {60, 111, 162, 213, 264, 315};
    aspect_ratios_ = {{2}, {2, 3}, {2, 3}, {2, 3}, {2}, {2}};   // 预选框尺寸：2 个正方形，边长为 min_size, max_size),
                                                                // 2 个或 4 个长方形，长宽为 sqrt(aspect_ratios_[i])*min_size, 1/sqrt(aspect_ratios_[i])*min_size
    nms_thresh_ = 0.5;
}

void SSDetection::SetDevice()
{
    torch::DeviceType device_type; 		// 设置 Device 类型

	// 判断是否支持 GPU 加速 torch::kCUDA or torch::kCPU
	if (torch::cuda::is_available())
	{
		cout << "CUDA available!" << endl;
		device_type = torch::kCUDA;
		cout << "Training on GPU." << endl;
	}
	else
	{
		device_type = torch::kCPU;
		cout << "Training on CPU." << endl;
	}

	torch::Device* device = new torch::Device(device_type);					// 定义设备类型 GPU or CPU

    device_ = device;
    std::cout << "设备配置成功！" << std::endl;
}

void SSDetection::LoadTracedModule(const std::string& weight)
{
    module_ = std::make_shared<torch::jit::script::Module>(torch::jit::load(weight)); // 加载模型权重
    std::cout << "Model load Successful!" << std::endl;
    
    module_->to(*device_);  // 模型转到 GPU or CPU 中去
    std::cout << "Device transform Successful!" << std::endl;
}

torch::Tensor SSDetection::Forward(const cv::Mat& image)
{
    cv::Mat input;
    cv::resize(image, input, cv::Size(net_size_, net_size_));   // 将图像尺寸调整为网络输入尺寸
    cv::cvtColor(input, input, cv::COLOR_BGR2RGB);              // 转换通道，将图像颜色信息格式由 BGR 转换为 RGB
                                                                // OpenCV 存储的颜色方式是 BGR，而 TorchScript 是 RGB

    // 将图像转化为 Tensor，随后导入模型进行预测
    torch::Tensor tensor_image = torch::from_blob(input.data, {1, input.rows, input.cols, 3}, torch::kByte);
    tensor_image = tensor_image.to(*device_);
    tensor_image = tensor_image.permute({0, 3, 1, 2});          // 翻转让通道是第二个维度 // Channels x Height x Width
    tensor_image = tensor_image.toType(torch::kFloat);
    // Tensor 标准化
    tensor_image[0][0].sub_(104);
    tensor_image[0][1].sub_(117);
    tensor_image[0][2].sub_(123);

/************************************************** 前向传播 **************************************************/

    auto start = std::chrono::high_resolution_clock::now();
    torch::Tensor output = module_->forward({tensor_image}).toTensor().squeeze(0);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    // std::cout << "前向传播 耗时 : " << duration.count() << " ms" << endl;

/************************************************** 预选框 **************************************************/

    start = std::chrono::high_resolution_clock::now();
    torch::Tensor prior_boxes = PriorBox(*device_);
    end = std::chrono::high_resolution_clock::now();
    duration = duration_cast<milliseconds>(end - start);
    // std::cout << "预选框  耗时: " << duration.count() << " ms" << endl;

/************************************************** 检测层 **************************************************/

    start = std::chrono::high_resolution_clock::now();
    torch::Tensor result = DetectionLayer(output, prior_boxes, nms_thresh_);
    end = std::chrono::high_resolution_clock::now();
    duration = duration_cast<milliseconds>(end - start);
    // std::cout << "检测层  耗时: " << duration.count() << " ms" << endl;

    return result;
}

torch::Tensor SSDetection::PriorBox (torch::Device device)
{
    int num_prior_box = 0;
    for(size_t i = 0; i < feature_maps_.size(); i++)
    {
        // 预选框的数量：像素点的数量 ×（不同比例的预选框的数量 × 2 + 2），每一层feature maps预选框的数量为 4, 6, 6, 6, 4, 4
        num_prior_box += feature_maps_[i] * feature_maps_[i] * (aspect_ratios_[i].size() * 2 + 2);
    }

    torch::Tensor prior_boxes = torch::empty({num_prior_box,4}).to(device);

    int64_t start = 0;
    int64_t end = 0;

    for(size_t i = 0; i < feature_maps_.size(); i++)
    {
        int num_anchor_layer = feature_maps_[i] * feature_maps_[i];                                             // 预选框中心点的数量
        int num_prior_boxes_layer = feature_maps_[i] * feature_maps_[i] * (aspect_ratios_[i].size() * 2 + 2);   // 预选框的数量
        auto feature_len = torch::arange(feature_maps_[i]).to(device);
        // meshgrid(x, y)
        std::vector<torch::Tensor> args = torch::meshgrid({feature_len, feature_len});
        // args[0] 中的每一行都是x中的一个元素  args[1] 中的每一列都是y中的一个元素
        // pytorch 中是反着来的
        torch::Tensor cy = args[0].contiguous().view({-1, 1});
        torch::Tensor cx = args[1].contiguous().view({-1, 1});

        // 这里必须要转换乘float，不转换是long
        cx = cx.toType(torch::kFloat);
        cy = cy.toType(torch::kFloat);
        float f_k = float(net_size_) / steps_[i];
        // unit center x,y
        //cx = (j + 0.5) / f_k  cy = (i + 0.5) / f_k
        //cx = cx.sub_(0.5).div_(torch::ones(cx.sizes()).to(device).mul_(f_k));
        cx.add_(0.5).div_(torch::ones({num_anchor_layer,1}).fill_(f_k).to(device));
        cy.add_(0.5).div_(torch::ones({num_anchor_layer,1}).fill_(f_k).to(device));

/************************************************** 预选框选择 **************************************************/

        float min_k = min_size_[i] / float(net_size_);  // 最小尺寸与网络输入尺寸的比例
        float max_k = max_size_[i] / float(net_size_);  // 最大尺寸与网络输入尺寸的比例

        // 2个正方形
        float s_k_min = min_k;
        torch::Tensor s_k_min_tensor = torch::ones({num_anchor_layer,1}).fill_(s_k_min).to(device);
        torch::Tensor prior_boxes_1 = torch::cat({cx, cy, s_k_min_tensor, s_k_min_tensor}, 1);

        float s_k_max = sqrt(min_k * max_k);
        torch::Tensor s_k_max_tensor = torch::ones({num_anchor_layer,1}).fill_(s_k_max).to(device);
        torch::Tensor prior_boxes_2 = torch::cat({cx, cy, s_k_max_tensor, s_k_max_tensor}, 1);

        torch::Tensor prior_boxes_layer = torch::cat({prior_boxes_1, prior_boxes_2}, 1);

        // 2 个或 4 个长方形
        for (size_t j = 0; j < aspect_ratios_[i].size(); j++)
        {
            float aspect_ratio = aspect_ratios_[i][j];
            float s_k_short = s_k_min * sqrt(aspect_ratio); // 预选框的宽
            float s_k_long = s_k_min / sqrt(aspect_ratio);  // 预选框的长
            torch::Tensor s_k_short_tensor = torch::ones({num_anchor_layer , 1}).fill_(s_k_short).to(device);
            torch::Tensor s_k_long_tensor = torch::ones({num_anchor_layer , 1}).fill_(s_k_long).to(device);

            torch::Tensor prior_boxes_3 = torch::cat({cx, cy, s_k_short_tensor, s_k_long_tensor}, 1);
            torch::Tensor prior_boxes_4 = torch::cat({cx, cy, s_k_long_tensor, s_k_short_tensor}, 1);

            prior_boxes_layer = torch::cat({prior_boxes_layer, prior_boxes_3, prior_boxes_4}, 1);
        }
        // 在 pytorch 中 priorbox 是先存放坐标(x,y)左右的 anchor_box，再存放下一个坐标
        prior_boxes_layer = prior_boxes_layer.contiguous().view({-1, 4});
        end = start + num_prior_boxes_layer;
        prior_boxes.slice(0, start, end) = prior_boxes_layer;
        start = end;
    }
    return prior_boxes;
}

torch::Tensor SSDetection::DetectionLayer(const torch::Tensor& output, const torch::Tensor& prior_boxes, float nms_thresh)
{
    nms_thresh_ = nms_thresh;
    
    // int num_priors = output.size(0);                                     // priors 个数
    int num_digit = output.size(1);                                         // 4 个 box + 1 个背景　+ num_classes
    torch::Tensor decode_loc = Decoder(output.slice(1, 0, 4), prior_boxes); // 预选框位置
    torch::Tensor conf = output.slice(1, 4, num_digit);                     // 背景和分类的置信度
    
    // 每个框，对应 num_classes + 1 中分数最大的，返回分数最大值和对应的下标idx
    std::tuple<torch::Tensor, torch::Tensor> max_classes = torch::max(output.slice(1, 4, num_digit), 1);
    // shape: num_priors
    auto max_conf = std::get<0>(max_classes).to(torch::kFloat32);
    // shape: num_priors
    auto max_conf_idx = std::get<1>(max_classes).to(torch::kFloat32);
    auto mask_conf_idx = torch::nonzero(max_conf_idx).squeeze();

    auto max_conf_t = max_conf.index_select(0, mask_conf_idx).unsqueeze(1);
    auto max_conf_idx_t = max_conf_idx.index_select(0, mask_conf_idx).unsqueeze(1);
    auto decode_loc_t = decode_loc.index_select(0, mask_conf_idx);

    torch::Tensor keep = nms(decode_loc_t, max_conf_t);

    decode_loc_t = torch::index_select(decode_loc_t, 0, keep);
    max_conf_t = torch::index_select(max_conf_t, 0, keep);
    // cout << max_conf_t << endl;
    max_conf_idx_t = torch::index_select(max_conf_idx_t, 0, keep);

    decode_loc_t = torch::cat({decode_loc_t, max_conf_t, max_conf_idx_t}, 1);

    return decode_loc_t;
}

torch::Tensor SSDetection::Decoder(torch::Tensor loc, const torch::Tensor& priors)
{
    // variance 0.1  0.2
    torch::Tensor box = torch::cat(
            { priors.slice(1,0,2) + loc.slice(1,0,2).mul(0.1) * priors.slice(1,2,4),
             priors.slice(1,2,4) * torch::exp(loc.slice(1,2,4).mul(0.2))} ,
            1);

    box.slice(1, 0, 2) -= box.slice(1, 2, 4).div(2);
    box.slice(1, 2, 4) += box.slice(1, 0, 2);
    return box.to(torch::kFloat32);
}

torch::Tensor SSDetection::nms(const torch::Tensor& decode_loc, const torch::Tensor& conf)
{
    torch::Tensor keep = torch::zeros({conf.size(0)}).to(torch::kLong).to(conf.device());
    torch::Tensor x1, y1, x2, y2; // 角坐标
    x1 = decode_loc.select(1, 0);
    y1 = decode_loc.select(1, 1);
    x2 = decode_loc.select(1, 2);
    y2 = decode_loc.select(1, 3);
    torch::Tensor area = torch::mul(x2- x1, y2 - y1);
    
    // 按照分数从大到小排序
    std::tuple<torch::Tensor,torch::Tensor> sort_ret = torch::sort(conf, 0, 1);
    torch::Tensor idx_set = std::get<1>(sort_ret).squeeze(1);;

    int count = 0;
    while (idx_set.numel() > 0)
    {
        auto i = idx_set[0];
        keep[count] = i;
        count++;
        if (idx_set.size(0) == 1)
            break;

        idx_set = torch::slice(idx_set, 0, 1, idx_set.size(0));

        torch::Tensor xx1 = torch::index_select(x1, 0, idx_set);
        torch::Tensor yy1 = torch::index_select(y1, 0, idx_set);
        torch::Tensor xx2 = torch::index_select(x2, 0, idx_set);
        torch::Tensor yy2 = torch::index_select(y2, 0, idx_set);

        torch::Tensor inter_rect_x1 = torch::max(xx1, x1[i]);
        torch::Tensor inter_rect_y1 = torch::max(yy1, y1[i]);
        torch::Tensor inter_rect_x2 = torch::min(xx2, x2[i]);
        torch::Tensor inter_rect_y2 = torch::min(yy2, y2[i]);

        // Intersection area
        torch::Tensor inter_area = torch::max(inter_rect_x2 - inter_rect_x1, torch::zeros(inter_rect_x2.sizes()).to(inter_rect_x2.device())) *
                                   torch::max(inter_rect_y2 - inter_rect_y1, torch::zeros(inter_rect_x2.sizes()).to(inter_rect_x2.device()));

        torch::Tensor union_area = torch::index_select(area, 0, idx_set) - inter_area + area[i];
        inter_area.div_(union_area);
        auto mask_idx = torch::nonzero(inter_area.mul_(inter_area < nms_thresh_)).squeeze(1);
        idx_set = torch::index_select(idx_set, 0, mask_idx);
    }
    
    return keep.slice(0, 0, count);
}

const int confidence_thresh = 0.99;
cv::Rect2f SSDetection::predict(cv::Mat& image)
{
    float width = image.cols;	// 输入图像的宽
	float height = image.rows;	// 输入图像的高
    
    // // 直方图均衡化，提高目标对比度
    // image = equalize_RGB(image);
	// 网络前向传播计算
	torch::Tensor detect_result = Forward(image);

	// x1, y1, x2, y2, score, label
	// Eigen 库
	detect_result.select(1, 0).mul_(width);
	detect_result.select(1, 1).mul_(height);
	detect_result.select(1, 2).mul_(width);
	detect_result.select(1, 3).mul_(height);

	detect_result = detect_result.cpu();
	// Return a `TensorAccessor` for CPU `Tensor`s. You have to specify scalar type and dimension
	auto detect_result_data = detect_result.accessor<float, 2>();

	float score = 0;	// 检测分数
	int numRoi = 0;		// 检测出的目标框的数量
    Rect2f roi;		    // 检测出的目标框

	for (int i = 0; i < detect_result.size(0); i++)
	{
		score = detect_result_data[i][4];
		if (score > confidence_thresh)
			numRoi++;
	}
    if (numRoi <= 0)
    {
        // cout << "重新检测" << endl;
        return roi;
    }
    // cout << "目标框的数量为" << numRoi << endl;

    string label = "";
    float max_box_score = 0.0;
	for (int i = 0; i < detect_result.size(0); i++)
	{
		score = detect_result_data[i][4];						// 检测分数
		label = VOC_CLASSES[detect_result_data[i][5]]; 	        // 检测标签

		if (score > confidence_thresh && label == "handrail")
		{
			// 检测出的目标框
            if (score > max_box_score)
            {
                max_box_score = score;
                roi.x = detect_result_data[i][0]; 									// 左上角坐标
			    roi.y = detect_result_data[i][1];
			    roi.width = detect_result_data[i][2] - detect_result_data[i][0]; 	// 宽和高
			    roi.height = detect_result_data[i][3] - detect_result_data[i][1];
                // cout << "label: " << label << "\t" << "score: " << score << endl;
                if (!isnormal(roi.x) || !isnormal(roi.x) || !isnormal(roi.width) || !isnormal(roi.height))
                {
                    roi.x = 0;
			        roi.y = 0;
			        roi.width = 0;
			        roi.height = 0;
                }
            }
			
		}
	}
    cout << "#current object: " << "roi.x: " << roi.x << "\troi.y: " << roi.y << "\troi.width: " << roi.width << "\troi.height: " << roi.height << endl;

    return roi;
}

cv::Mat SSDetection::equalize_RGB(cv::Mat& image)
{
    cv::Mat hsv;
    cv::Mat dst;
     
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> channels;
    cv::split(hsv, channels);
     
    // 均衡化 v 通道
    cv::equalizeHist(channels[2], channels[2]);
    cv::merge(channels, hsv);
     
    cv::cvtColor(hsv, dst, cv::COLOR_HSV2BGR);
    return dst;
}