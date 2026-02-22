#ifndef JETSON_DETECT_YOLOV8_HPP
#define JETSON_DETECT_YOLOV8_HPP
#include "NvInferRuntime.h"
#include "NvInferRuntimeCommon.h"
#include "NvInferPlugin.h"
#include "common.hpp"
#include "fstream"
#include <memory>
#include <unordered_map>
#include <vector>

// Add OpenCV includes for NMS
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include "sl/Camera.hpp"
#include "usv_interfaces/msg/zbbox.hpp"
#include "usv_interfaces/msg/zbbox_array.hpp"

using namespace det;

class YOLOv8 {
public:
    explicit YOLOv8(const std::string& engine_file_path, double threshold, rclcpp::Logger logger_param);
    ~YOLOv8();

    rclcpp::Logger logger;

    void make_pipe(bool warmup = true);
    void copy_from_Mat(const cv::Mat& image);
    void copy_from_Mat(const cv::Mat& image, const cv::Size& size);
    void letterbox(const cv::Mat& image, cv::Mat& out, const cv::Size& size);
    void infer();
    usv_interfaces::msg::ZbboxArray postprocess();
    void draw_objects(
        const cv::Mat& image,
        cv::Mat& res,
        const usv_interfaces::msg::ZbboxArray& objs,
        const std::vector<std::string>& CLASS_NAMES,
        const std::vector<std::vector<unsigned int>>& COLORS);

    double threshold = 0.0;
    // NMS Threshold for overlapping boxes
    float nms_threshold = 0.45f; 
    PreParam pparam;

private:
    // TensorRT components
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t stream = nullptr;
    Logger gLogger{nvinfer1::ILogger::Severity::kERROR};
    
    // Name-based buffers
    struct TensorInfo {
        std::string name;
        nvinfer1::DataType dataType;
        nvinfer1::Dims dims;
        size_t size; // Total bytes
        size_t elementSize; // Size of one element in bytes
        bool isInput;
        void* deviceBuffer = nullptr;
        void* hostBuffer = nullptr;
    };
    
    std::unordered_map<std::string, TensorInfo> tensors;
    std::vector<void*> deviceBuffers;
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;

    // Name-based inference properties
    std::string inputBlobName;
    std::string outputBlobName; // Replaced the 4 specific tensor names with a single output
};

YOLOv8::YOLOv8(const std::string& engine_file_path, double threshold_param, rclcpp::Logger logger_param) 
    : logger(logger_param), threshold(threshold_param)
{
    // Load engine file
    std::ifstream file(engine_file_path, std::ios::binary);
    if (!file.good()) {
        RCLCPP_ERROR(this->logger, "Failed to open engine file: %s", engine_file_path.c_str());
        throw std::runtime_error("Failed to open engine file");
    }
    
    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<char> trtModelStream(size);
    if (!trtModelStream.data()) {
        RCLCPP_ERROR(this->logger, "Failed to allocate memory for engine file");
        throw std::runtime_error("Failed to allocate memory for engine file");
    }
    
    file.read(trtModelStream.data(), size);
    file.close();

    // Initialize TensorRT
    initLibNvInferPlugins(&this->gLogger, "");
    
    // Create runtime
    this->runtime = nvinfer1::createInferRuntime(this->gLogger);
    if (!this->runtime) {
        RCLCPP_ERROR(this->logger, "Failed to create TensorRT Runtime");
        throw std::runtime_error("Failed to create TensorRT Runtime");
    }

    // Deserialize engine
    this->engine = this->runtime->deserializeCudaEngine(trtModelStream.data(), size);
    if (!this->engine) {
        RCLCPP_ERROR(this->logger, "Failed to deserialize CUDA engine");
        if (this->runtime) {
            delete this->runtime;
            this->runtime = nullptr;
        }
        throw std::runtime_error("Failed to create TensorRT engine");
    }

    // Create execution context
    this->context = this->engine->createExecutionContext();
    if (!this->context) {
        RCLCPP_ERROR(this->logger, "Failed to create execution context");
        if (this->engine) { delete this->engine; this->engine = nullptr; }
        if (this->runtime) { delete this->runtime; this->runtime = nullptr; }
        throw std::runtime_error("Failed to create execution context");
    }

    // Create CUDA stream
    cudaError_t cudaErr = cudaStreamCreate(&this->stream);
    if (cudaErr != cudaSuccess) {
        RCLCPP_ERROR(this->logger, "Failed to create CUDA stream: %s", cudaGetErrorString(cudaErr));
        if (this->context) { delete this->context; this->context = nullptr; }
        if (this->engine) { delete this->engine; this->engine = nullptr; }
        if (this->runtime) { delete this->runtime; this->runtime = nullptr; }
        throw std::runtime_error("Failed to create CUDA stream");
    }

    // Get tensor information
    int numBindings = this->engine->getNbIOTensors();
    for (int i = 0; i < numBindings; i++) {
        const char* tensorName = this->engine->getIOTensorName(i);
        TensorInfo tensorInfo;
        tensorInfo.name = tensorName;
        tensorInfo.isInput = this->engine->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT;
        tensorInfo.dataType = this->engine->getTensorDataType(tensorName);
        
        switch (tensorInfo.dataType) {
            case nvinfer1::DataType::kFLOAT: tensorInfo.elementSize = 4; break;
            case nvinfer1::DataType::kHALF: tensorInfo.elementSize = 2; break;
            case nvinfer1::DataType::kINT8: tensorInfo.elementSize = 1; break;
            case nvinfer1::DataType::kINT32: tensorInfo.elementSize = 4; break;
            case nvinfer1::DataType::kBOOL: tensorInfo.elementSize = 1; break;
            default: tensorInfo.elementSize = 4; break;
        }
        
        tensorInfo.dims = this->engine->getTensorShape(tensorName);

        if (tensorInfo.isInput) {
            inputNames.push_back(tensorName);
            inputBlobName = tensorName;
        } else {
            outputNames.push_back(tensorName);
            outputBlobName = tensorName; // Capture the standard output tensor
        }
        
        size_t totalSize = 1;
        for (int d = 0; d < tensorInfo.dims.nbDims; d++) {
            totalSize *= tensorInfo.dims.d[d];
        }
        tensorInfo.size = totalSize * tensorInfo.elementSize;
        
        tensors[tensorName] = tensorInfo;
    }
    
    if (outputNames.empty()) {
        RCLCPP_ERROR(this->logger, "No output tensors found in model");
        throw std::runtime_error("Model has no output tensors");
    }
    
    RCLCPP_INFO(this->logger, "TensorRT engine loaded successfully");
    RCLCPP_INFO(this->logger, "  Input: %s", inputBlobName.c_str());
    RCLCPP_INFO(this->logger, "  Output: %s", outputBlobName.c_str());
}

YOLOv8::~YOLOv8()
{
    if (this->stream) { cudaStreamDestroy(this->stream); this->stream = nullptr; }
    
    for (auto& pair : tensors) {
        auto& tensor = pair.second;
        if (tensor.deviceBuffer) { cudaFree(tensor.deviceBuffer); tensor.deviceBuffer = nullptr; }
        if (tensor.hostBuffer) { cudaFreeHost(tensor.hostBuffer); tensor.hostBuffer = nullptr; }
    }
    
    if (this->context) { delete this->context; this->context = nullptr; }
    if (this->engine) { delete this->engine; this->engine = nullptr; }
    if (this->runtime) { delete this->runtime; this->runtime = nullptr; }
}

void YOLOv8::make_pipe(bool warmup)
{
    for (auto& pair : tensors) {
        auto& tensor = pair.second;
        cudaError_t err = cudaMalloc(&tensor.deviceBuffer, tensor.size);
        if (err != cudaSuccess) {
            RCLCPP_ERROR(this->logger, "CUDA malloc failed for tensor %s", tensor.name.c_str());
            throw std::runtime_error("CUDA malloc failed");
        }
        
        if (!tensor.isInput) {
            err = cudaHostAlloc(&tensor.hostBuffer, tensor.size, cudaHostAllocDefault);
            if (err != cudaSuccess) {
                RCLCPP_ERROR(this->logger, "CUDA host alloc failed for tensor %s", tensor.name.c_str());
                throw std::runtime_error("CUDA host alloc failed");
            }
        }
        deviceBuffers.push_back(tensor.deviceBuffer);
    }

    if (warmup) {
        RCLCPP_INFO(this->logger, "Warming up model...");
        for (int i = 0; i < 10; i++) {
            void* dummyInput = malloc(tensors[inputBlobName].size);
            memset(dummyInput, 0, tensors[inputBlobName].size);
            cudaMemcpyAsync(tensors[inputBlobName].deviceBuffer, dummyInput, tensors[inputBlobName].size, cudaMemcpyHostToDevice, this->stream);
            free(dummyInput);
            this->infer();
        }
        RCLCPP_INFO(this->logger, "Model warmup completed");
    }
}

void YOLOv8::letterbox(const cv::Mat& image, cv::Mat& out, const cv::Size& size)
{
    const float inp_h  = size.height;
    const float inp_w  = size.width;
    float       height = image.rows;
    float       width  = image.cols;

    float r    = std::min(inp_h / height, inp_w / width);
    int   padw = std::round(width * r);
    int   padh = std::round(height * r);

    cv::Mat tmp;
    if ((int)width != padw || (int)height != padh) {
        cv::resize(image, tmp, cv::Size(padw, padh));
    }
    else {
        tmp = image.clone();
    }

    float dw = inp_w - padw;
    float dh = inp_h - padh;

    dw /= 2.0f;
    dh /= 2.0f;
    int top    = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left   = int(std::round(dw - 0.1f));
    int right  = int(std::round(dw + 0.1f));

    cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT, {114, 114, 114});

    cv::dnn::blobFromImage(tmp, out, 1 / 255.f, cv::Size(), cv::Scalar(0, 0, 0), true, false, CV_32F);
    this->pparam.ratio  = 1 / r;
    this->pparam.dw     = dw;
    this->pparam.dh     = dh;
    this->pparam.height = height;
    this->pparam.width  = width;
}

void YOLOv8::copy_from_Mat(const cv::Mat& image)
{
    auto& inputTensor = tensors[inputBlobName];
    cv::Mat nchw;
    int height = inputTensor.dims.d[2]; 
    int width = inputTensor.dims.d[3];
    cv::Size size{width, height};
    
    this->letterbox(image, nchw, size);
    
    if (this->engine->getTensorShape(inputBlobName.c_str()).d[0] == -1) {
        nvinfer1::Dims4 inputDims{1, 3, height, width};
        this->context->setInputShape(inputBlobName.c_str(), inputDims);
    }
    
    cudaMemcpyAsync(inputTensor.deviceBuffer, nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, this->stream);
}

void YOLOv8::copy_from_Mat(const cv::Mat& image, const cv::Size& size)
{
    auto& inputTensor = tensors[inputBlobName];
    cv::Mat nchw;
    
    this->letterbox(image, nchw, size);
    
    if (this->engine->getTensorShape(inputBlobName.c_str()).d[0] == -1) {
        nvinfer1::Dims4 inputDims{1, 3, size.height, size.width};
        this->context->setInputShape(inputBlobName.c_str(), inputDims);
    }
    
    cudaMemcpyAsync(inputTensor.deviceBuffer, nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, this->stream);
}

void YOLOv8::infer()
{
    for (const auto& name : inputNames) {
        this->context->setTensorAddress(name.c_str(), tensors[name].deviceBuffer);
    }

    for (const auto& name : outputNames) {
        this->context->setTensorAddress(name.c_str(), tensors[name].deviceBuffer);
    }

    this->context->enqueueV3(this->stream);

    for (const auto& name : outputNames) {
        auto& tensor = tensors[name];
        cudaMemcpyAsync(tensor.hostBuffer, tensor.deviceBuffer, tensor.size, cudaMemcpyDeviceToHost, this->stream);
    }

    cudaStreamSynchronize(this->stream);
}


usv_interfaces::msg::ZbboxArray YOLOv8::postprocess()
{
    usv_interfaces::msg::ZbboxArray arr;

    // Grab the single output tensor
    float* output = static_cast<float*>(tensors[outputBlobName].hostBuffer);
    auto dims = tensors[outputBlobName].dims;
    
    // YOLOv8 Standard Output Shape: [1, num_channels, num_anchors]
    // Where num_channels = 4 (for boxes) + num_classes
    int num_channels = dims.d[1];
    int num_anchors = dims.d[2];
    int num_classes = num_channels - 4;

    auto& dw       = this->pparam.dw;
    auto& dh       = this->pparam.dh;
    auto& width    = this->pparam.width;
    auto& height   = this->pparam.height;
    auto& ratio    = this->pparam.ratio;

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;

    // 1. Parse the raw tensor and filter by confidence
    // The memory is laid out channel-major, meaning all 8400 X's come first, then 8400 Y's, etc.
    for (int a = 0; a < num_anchors; ++a) {
        float max_score = 0.0f;
        int max_class_idx = -1;
        
        // Find the class with the highest score for this anchor
        for (int c = 0; c < num_classes; ++c) {
            float score = output[(4 + c) * num_anchors + a];
            if (score > max_score) {
                max_score = score;
                max_class_idx = c;
            }
        }

        if (max_score >= this->threshold) {
            float cx = output[0 * num_anchors + a]; // Center X
            float cy = output[1 * num_anchors + a]; // Center Y
            float w  = output[2 * num_anchors + a]; // Width
            float h  = output[3 * num_anchors + a]; // Height

            // Convert to top-left corner format for OpenCV
            int left = static_cast<int>(cx - w / 2.0f);
            int top  = static_cast<int>(cy - h / 2.0f);

            boxes.push_back(cv::Rect(left, top, static_cast<int>(w), static_cast<int>(h)));
            scores.push_back(max_score);
            class_ids.push_back(max_class_idx);
        }
    }

    // 2. Perform Non-Maximum Suppression (NMS)
    std::vector<int> nms_indices;
    cv::dnn::NMSBoxes(boxes, scores, this->threshold, this->nms_threshold, nms_indices);

    // 3. Convert surviving boxes to ROS messages and scale back to original image size
    for (int idx : nms_indices) {
        cv::Rect box = boxes[idx];
        
        float x0 = box.x - dw;
        float y0 = box.y - dh;
        float x1 = (box.x + box.width) - dw;
        float y1 = (box.y + box.height) - dh;

        x0 = clamp(x0 * ratio, 0.f, width);
        y0 = clamp(y0 * ratio, 0.f, height);
        x1 = clamp(x1 * ratio, 0.f, width);
        y1 = clamp(y1 * ratio, 0.f, height);

        usv_interfaces::msg::Zbbox obj;
        obj.uuid = sl::generate_unique_id();
        obj.prob = scores[idx];
        obj.label = class_ids[idx];
        
        obj.x0 = x0;
        obj.y0 = y0;
        obj.x1 = x1;
        obj.y1 = y1;

        arr.boxes.push_back(obj);
    }

    return arr;
}

void YOLOv8::draw_objects(
    const cv::Mat& image,
    cv::Mat& res,
    const usv_interfaces::msg::ZbboxArray& objs,
    const std::vector<std::string>& CLASS_NAMES,
    const std::vector<std::vector<unsigned int>>& COLORS)
{
    res = image.clone();

    for (const auto& obj : objs.boxes) {
        if (obj.label >= CLASS_NAMES.size() || obj.label >= COLORS.size()) {
            continue;
        }

        cv::Scalar color(COLORS[obj.label][0], COLORS[obj.label][1], COLORS[obj.label][2]);

        int rect_x = static_cast<int>(obj.x0);
        int rect_y = static_cast<int>(obj.y0);
        int rect_width = static_cast<int>(obj.x1 - obj.x0);
        int rect_height = static_cast<int>(obj.y1 - obj.y0);
        cv::Rect rect(rect_x, rect_y, rect_width, rect_height);

        cv::rectangle(res, rect, color, 2);

        std::string label_text = cv::format("%s %d%%", CLASS_NAMES[obj.label].c_str(), (int)(obj.prob*100.0));

        std::string label_text = cv::format("%s %d%%",
            CLASS_NAMES[obj.label].c_str(), (int)(obj.prob*100.0));

        // Calculate text size and position
        int baseline = 0;
        cv::Size label_size = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseline);

        int x = rect_x;
        int y = rect_y;
        y = std::min(y + 1, res.rows - label_size.height - baseline);

        cv::rectangle(res, cv::Rect(x, y, label_size.width, label_size.height + baseline), cv::Scalar(0, 0, 255), -1);
        cv::putText(res, label_text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
    }
}

#endif  // JETSON_DETECT_YOLOV8_HPP