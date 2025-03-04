#ifndef YOLO_TENSORRT_HPP
#define YOLO_TENSORRT_HPP

// tensorrt
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvInferRuntime.h"

// zedsdk
#include "sl/Camera.hpp"

// ros interfaces
#include "usv_interfaces/msg/zbbox.hpp"
#include "usv_interfaces/msg/zbbox_array.hpp"

// extra
#include "common.hpp"
#include "fstream"
#include <memory>

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
    
    static void draw_objects(const cv::Mat& image,
                           cv::Mat& res,
                           const usv_interfaces::msg::ZbboxArray& objs,
                           const std::vector<std::string>& CLASS_NAMES,
                           const std::vector<std::vector<unsigned int>>& COLORS);

    int num_bindings;
    int num_inputs = 0;
    int num_outputs = 0;
    std::vector<Binding> input_bindings;
    std::vector<Binding> output_bindings;
    std::vector<void*> host_ptrs;
    std::vector<void*> device_ptrs;
    double threshold = 0.0;
    PreParam pparam;

private:
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    cudaStream_t stream = nullptr;
    Logger gLogger{nvinfer1::ILogger::Severity::kERROR};
};

YOLOv8::YOLOv8(const std::string& engine_file_path, double threshold_param, rclcpp::Logger logger_param) 
    : logger(logger_param), threshold(threshold_param) {
    
    std::ifstream file(engine_file_path, std::ios::binary);
    if (!file.good()) {
        throw std::runtime_error("Failed to open engine file: " + engine_file_path);
    }

    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<char> trtModelStream(size);
    if (!file.read(trtModelStream.data(), size)) {
        throw std::runtime_error("Failed to read engine file");
    }
    file.close();

    initLibNvInferPlugins(&this->gLogger, "");
    
    this->runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(this->gLogger));
    if (!this->runtime) {
        throw std::runtime_error("failed to create tensorrt runtime");
    }

    this->engine = std::unique_ptr<nvinfer1::ICudaEngine>(
        this->runtime->deserializeCudaEngine(trtModelStream.data(), size));
    if (!this->engine) {
        throw std::runtime_error("failed to create CUDA engine");
    }

    this->context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (!this->context) {
        throw std::runtime_error("failed to create execution context");
    }

    if (cudaStreamCreate(&this->stream) != cudaSuccess) {
        throw std::runtime_error("failed to create CUDA stream");
    }

    // Initialize bindings
    this->num_bindings = this->engine->getNbIOTensors();
    
    for (int i = 0; i < this->num_bindings; ++i) {
        Binding binding;
        const char* tensor_name = this->engine->getIOTensorName(i);
        nvinfer1::Dims dims = this->engine->getTensorShape(tensor_name);
        nvinfer1::DataType dtype = this->engine->getTensorDataType(tensor_name);
        
        binding.name = tensor_name;
        binding.dsize = type_to_size(dtype);

        if (this->engine->getTensorIOMode(tensor_name) == nvinfer1::TensorIOMode::kINPUT) {
            this->num_inputs++;
            // For input tensors, get the max profile dimensions
            dims = this->engine->getProfileShape(tensor_name, 0, nvinfer1::OptProfileSelector::kMAX);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->input_bindings.push_back(binding);
            
            // Set the input shape for the execution context
            if (!this->context->setInputShape(tensor_name, dims)) {
                throw std::runtime_error("Failed to set input shape for tensor: " + std::string(tensor_name));
            }
        } else {
            this->num_outputs++;
            dims = this->context->getTensorShape(tensor_name);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->output_bindings.push_back(binding);
        }
    }
}

YOLOv8::~YOLOv8() {
    if (stream) {
        cudaStreamDestroy(stream);
    }
    
    for (auto& ptr : device_ptrs) {
        if (ptr) {
            cudaFree(ptr);
        }
    }
    
    for (auto& ptr : host_ptrs) {
        if (ptr) {
            cudaFreeHost(ptr);
        }
    }
}

void YOLOv8::make_pipe(bool warmup) {
    // Allocate device memory for inputs
    for (const auto& binding : input_bindings) {
        void* d_ptr = nullptr;
        if (cudaMalloc(&d_ptr, binding.size * binding.dsize) != cudaSuccess) {
            throw std::runtime_error("Failed to allocate device memory for input");
        }
        device_ptrs.push_back(d_ptr);
    }

    // Allocate device and host memory for outputs
    for (const auto& binding : output_bindings) {
        void *d_ptr = nullptr, *h_ptr = nullptr;
        size_t size = binding.size * binding.dsize;
        
        if (cudaMalloc(&d_ptr, size) != cudaSuccess) {
            throw std::runtime_error("Failed to allocate device memory for output");
        }
        if (cudaHostAlloc(&h_ptr, size, cudaHostAllocDefault) != cudaSuccess) {
            cudaFree(d_ptr);
            throw std::runtime_error("Failed to allocate host memory for output");
        }
        
        device_ptrs.push_back(d_ptr);
        host_ptrs.push_back(h_ptr);
    }

    if (warmup) {
        std::vector<float> dummy_input(input_bindings[0].size, 0.0f);
        for (int i = 0; i < 10; i++) {
            cudaMemcpyAsync(device_ptrs[0], dummy_input.data(), 
                           dummy_input.size() * sizeof(float), 
                           cudaMemcpyHostToDevice, stream);
            infer();
        }
    }
}

void YOLOv8::draw_objects(
    const cv::Mat& image,
    cv::Mat& res,
    const usv_interfaces::msg::ZbboxArray& objs,
    const std::vector<std::string>& CLASS_NAMES,
    const std::vector<std::vector<unsigned int>>& COLORS)
{
    // Create a copy of the input image for drawing
    res = image.clone();

    // Iterate over each detection in the ZbboxArray message
    for (const auto& obj : objs.boxes) {
        // Ensure label index is within bounds
        if (obj.label >= CLASS_NAMES.size() || obj.label >= COLORS.size()) {
            RCLCPP_WARN(rclcpp::get_logger("YOLOv8"),
                "Invalid label index: %d. Skipping drawing.", obj.label);
            continue;
        }

        // Create color from the predefined color array
        cv::Scalar color(
            COLORS[obj.label][0],
            COLORS[obj.label][1],
            COLORS[obj.label][2]
        );

        // Create rectangle using x0, y0, x1, and y1.
        int rect_x = static_cast<int>(obj.x0);
        int rect_y = static_cast<int>(obj.y0);
        int rect_width = static_cast<int>(obj.x1 - obj.x0);
        int rect_height = static_cast<int>(obj.y1 - obj.y0);
        cv::Rect rect(rect_x, rect_y, rect_width, rect_height);

        // Draw bounding box
        cv::rectangle(res, rect, color, 2);

        // Prepare label text with confidence score
        std::string label_text = cv::format("%s %.1f%%",
            CLASS_NAMES[obj.label].c_str(), obj.prob * 100);

        // Calculate text size and position
        int baseline = 0;
        cv::Size label_size = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseline);

        // Calculate label background position
        int x = rect_x;
        int y = rect_y;
        y = std::min(y + 1, res.rows - label_size.height - baseline);

        // Draw label background
        cv::rectangle(res,
            cv::Rect(x, y, label_size.width, label_size.height + baseline),
            cv::Scalar(0, 0, 255), -1);

        // Draw label text
        cv::putText(res, label_text,
            cv::Point(x, y + label_size.height),
            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
    }
}

void YOLOv8::letterbox(const cv::Mat& image, cv::Mat& out, const cv::Size& size) {
    const float inp_h = size.height;
    const float inp_w = size.width;
    float height = image.rows;
    float width = image.cols;
    
    float r = std::min(inp_h / height, inp_w / width);
    int padw = std::round(width * r);
    int padh = std::round(height * r);
    
    cv::Mat tmp;
    if ((int)width != padw || (int)height != padh) {
        cv::resize(image, tmp, cv::Size(padw, padh));
    } else {
        tmp = image.clone();
    }
    
    float dw = inp_w - padw;
    float dh = inp_h - padh;
    dw /= 2.0f;
    dh /= 2.0f;
    
    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));
    
    cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, 
                       cv::BORDER_CONSTANT, {114, 114, 114});
    
    cv::dnn::blobFromImage(tmp, out, 1 / 255.f, cv::Size(), 
                          cv::Scalar(0, 0, 0), true, false, CV_32F);
    
    pparam.ratio = 1 / r;
    pparam.dw = dw;
    pparam.dh = dh;
    pparam.height = height;
    pparam.width = width;
}

void YOLOv8::copy_from_Mat(const cv::Mat& image) {
    cv::Mat nchw;
    const auto& in_binding = input_bindings[0];
    auto width = in_binding.dims.d[3];
    auto height = in_binding.dims.d[2];
    cv::Size size{width, height};
    
    letterbox(image, nchw, size);
    
    const char* tensor_name = engine->getIOTensorName(0);
    if (!context->setInputShape(tensor_name, nvinfer1::Dims4{1, 3, height, width})) {
        throw std::runtime_error("Failed to set input shape");
    }
    
    cudaMemcpyAsync(device_ptrs[0], nchw.ptr<float>(), 
                    nchw.total() * nchw.elemSize(), 
                    cudaMemcpyHostToDevice, stream);
}

void YOLOv8::copy_from_Mat(const cv::Mat& image, const cv::Size& size) {
    cv::Mat nchw;
    letterbox(image, nchw, size);
    
    const char* tensor_name = engine->getIOTensorName(0);
    if (!context->setInputShape(tensor_name, nvinfer1::Dims4{1, 3, size.height, size.width})) {
        throw std::runtime_error("Failed to set input shape");
    }
    
    cudaMemcpyAsync(device_ptrs[0], nchw.ptr<float>(), 
                    nchw.total() * nchw.elemSize(), 
                    cudaMemcpyHostToDevice, stream);
}

void YOLOv8::infer() {
    // Set tensor addresses for all IO tensors
    for (int i = 0; i < num_inputs; i++) {
        const char* tensor_name = input_bindings[i].name.c_str();
        if (!context->setTensorAddress(tensor_name, device_ptrs[i])) {
            throw std::runtime_error("Failed to set input tensor address");
        }
    }
    
    for (int i = 0; i < num_outputs; i++) {
        const char* tensor_name = output_bindings[i].name.c_str();
        if (!context->setTensorAddress(tensor_name, device_ptrs[i + num_inputs])) {
            throw std::runtime_error("Failed to set output tensor address");
        }
    }

    // Enqueue inference work
    if (!context->enqueueV3(stream)) {
        throw std::runtime_error("Failed to enqueue inference work");
    }

    // Copy output data back to host
    for (int i = 0; i < num_outputs; i++) {
        size_t osize = output_bindings[i].size * output_bindings[i].dsize;
        cudaMemcpyAsync(host_ptrs[i], device_ptrs[i + num_inputs], 
                       osize, cudaMemcpyDeviceToHost, stream);
    }
    
    cudaStreamSynchronize(stream);
}

usv_interfaces::msg::ZbboxArray YOLOv8::postprocess() {
    usv_interfaces::msg::ZbboxArray arr;
    
    int* num_dets = static_cast<int*>(host_ptrs[0]);
    float* boxes = static_cast<float*>(host_ptrs[1]);
    float* scores = static_cast<float*>(host_ptrs[2]);
    int* labels = static_cast<int*>(host_ptrs[3]);
    
    const auto& dw = pparam.dw;
    const auto& dh = pparam.dh;
    const auto& width = pparam.width;
    const auto& height = pparam.height;
    const auto& ratio = pparam.ratio;
    
    RCLCPP_DEBUG(logger, "Number of detections: %d", num_dets[0]);
    RCLCPP_DEBUG(logger, "Confidence threshold: %f", threshold);
    
    for (int i = 0; i < num_dets[0]; i++) {
        if (scores[i] <= threshold) {
            continue;
        }
        
        float* ptr = boxes + i * 4;
        float x0 = (*ptr++ - dw) * ratio;
        float y0 = (*ptr++ - dh) * ratio;
        float x1 = (*ptr++ - dw) * ratio;
        float y1 = (*ptr - dh) * ratio;
        
        x0 = clamp(x0, 0.f, width);
        y0 = clamp(y0, 0.f, height);
        x1 = clamp(x1, 0.f, width);
        y1 = clamp(y1, 0.f, height);
        
        usv_interfaces::msg::Zbbox obj;
        obj.uuid = sl::generate_unique_id();
        obj.prob = scores[i];
        obj.label = labels[i];
        obj.x0 = x0;
        obj.y0 = y0;
        obj.x1 = x1;
        obj.y1 = y1;
        
        arr.boxes.push_back(obj);
    }

    RCLCPP_DEBUG(this->logger, "pushing back size: %d", arr.boxes.size());
    
    return arr;
}
#endif
