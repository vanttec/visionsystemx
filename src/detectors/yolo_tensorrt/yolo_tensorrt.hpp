#ifndef JETSON_DETECT_YOLOV8_HPP
#define JETSON_DETECT_YOLOV8_HPP
#include "NvInferRuntime.h"
#include "NvInferRuntimeCommon.h"
#include "NvInferPlugin.h"
#include "common.hpp"
#include "fstream"
#include <memory>
#include <unordered_map>

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
    std::string numDetectionsName;
    std::string boxesName;
    std::string scoresName;
    std::string classesName;
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
            // Handle cleanup if engine creation failed
            delete this->runtime;
            this->runtime = nullptr;
        }
        throw std::runtime_error("Failed to create TensorRT engine");
    }

    // Create execution context
    this->context = this->engine->createExecutionContext();
    if (!this->context) {
        RCLCPP_ERROR(this->logger, "Failed to create execution context");
        // Clean up if context creation failed
        if (this->engine) {
            delete this->engine;
            this->engine = nullptr;
        }
        if (this->runtime) {
            delete this->runtime;
            this->runtime = nullptr;
        }
        throw std::runtime_error("Failed to create execution context");
    }

    // Create CUDA stream
    cudaError_t cudaErr = cudaStreamCreate(&this->stream);
    if (cudaErr != cudaSuccess) {
        RCLCPP_ERROR(this->logger, "Failed to create CUDA stream: %s", cudaGetErrorString(cudaErr));
        // Clean up if stream creation failed
        if (this->context) {
            delete this->context;
            this->context = nullptr;
        }
        if (this->engine) {
            delete this->engine;
            this->engine = nullptr;
        }
        if (this->runtime) {
            delete this->runtime;
            this->runtime = nullptr;
        }
        throw std::runtime_error("Failed to create CUDA stream");
    }

    // Get tensor information using name-based approach
    int numBindings = this->engine->getNbIOTensors();
    for (int i = 0; i < numBindings; i++) {
        const char* tensorName = this->engine->getIOTensorName(i);
        TensorInfo tensorInfo;
        tensorInfo.name = tensorName;
        tensorInfo.isInput = this->engine->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT;
        tensorInfo.dataType = this->engine->getTensorDataType(tensorName);
        
        // Get element size
        switch (tensorInfo.dataType) {
            case nvinfer1::DataType::kFLOAT: tensorInfo.elementSize = 4; break;
            case nvinfer1::DataType::kHALF: tensorInfo.elementSize = 2; break;
            case nvinfer1::DataType::kINT8: tensorInfo.elementSize = 1; break;
            case nvinfer1::DataType::kINT32: tensorInfo.elementSize = 4; break;
            case nvinfer1::DataType::kBOOL: tensorInfo.elementSize = 1; break;
            default: tensorInfo.elementSize = 4; break;
        }
        
        if (tensorInfo.isInput) {
            // For input tensors, get the dimensions
            tensorInfo.dims = this->engine->getTensorShape(tensorName);
            inputNames.push_back(tensorName);
            
            // Save the input name for later use
            inputBlobName = tensorName;
        } else {
            // For output tensors, save their names based on expected content
            outputNames.push_back(tensorName);
            
            // Try to identify output tensors by name pattern or shape
            std::string name = tensorName;
            if (name.find("num_dets") != std::string::npos || name.find("count") != std::string::npos) {
                numDetectionsName = name;
            } else if (name.find("box") != std::string::npos || name.find("bbox") != std::string::npos) {
                boxesName = name;
            } else if (name.find("score") != std::string::npos || name.find("conf") != std::string::npos) {
                scoresName = name;
            } else if (name.find("class") != std::string::npos || name.find("label") != std::string::npos) {
                classesName = name;
            }
            
            // Get tensor dims
            tensorInfo.dims = this->engine->getTensorShape(tensorName);
        }
        
        // Calculate total size
        size_t totalSize = 1;
        for (int d = 0; d < tensorInfo.dims.nbDims; d++) {
            totalSize *= tensorInfo.dims.d[d];
        }
        tensorInfo.size = totalSize * tensorInfo.elementSize;
        
        // Store tensor info
        tensors[tensorName] = tensorInfo;
    }
    
    // Verify we identified all needed output tensors
    if (numDetectionsName.empty() || boxesName.empty() || scoresName.empty() || classesName.empty()) {
        RCLCPP_WARN(this->logger, "Could not identify all output tensors by name pattern. Using order-based assignment.");
        if (outputNames.size() >= 4) {
            numDetectionsName = outputNames[0];
            boxesName = outputNames[1];
            scoresName = outputNames[2];
            classesName = outputNames[3];
        } else {
            RCLCPP_ERROR(this->logger, "Not enough output tensors in model");
            throw std::runtime_error("Model has insufficient output tensors");
        }
    }
    
    RCLCPP_INFO(this->logger, "TensorRT engine loaded successfully");
    RCLCPP_INFO(this->logger, "  Input: %s", inputBlobName.c_str());
    RCLCPP_INFO(this->logger, "  Outputs: num_dets=%s, boxes=%s, scores=%s, classes=%s", 
               numDetectionsName.c_str(), boxesName.c_str(), scoresName.c_str(), classesName.c_str());
}

YOLOv8::~YOLOv8()
{
    // Free CUDA resources
    if (this->stream) {
        cudaStreamDestroy(this->stream);
        this->stream = nullptr;
    }
    
    // Free device and host buffers
    for (auto& pair : tensors) {
        auto& tensor = pair.second;
        if (tensor.deviceBuffer) {
            cudaFree(tensor.deviceBuffer);
            tensor.deviceBuffer = nullptr;
        }
        if (tensor.hostBuffer) {
            cudaFreeHost(tensor.hostBuffer);
            tensor.hostBuffer = nullptr;
        }
    }
    
    // Clean up TensorRT objects
    if (this->context) {
        delete this->context;
        this->context = nullptr;
    }
    
    if (this->engine) {
        delete this->engine;
        this->engine = nullptr;
    }
    
    if (this->runtime) {
        delete this->runtime;
        this->runtime = nullptr;
    }
}

void YOLOv8::make_pipe(bool warmup)
{
    // Allocate device and host memory for each tensor
    for (auto& pair : tensors) {
        auto& tensor = pair.second;
        
        // Allocate device memory
        cudaError_t err = cudaMalloc(&tensor.deviceBuffer, tensor.size);
        if (err != cudaSuccess) {
            RCLCPP_ERROR(this->logger, "CUDA malloc failed for tensor %s: %s", 
                         tensor.name.c_str(), cudaGetErrorString(err));
            throw std::runtime_error("CUDA malloc failed");
        }
        
        // For output tensors, also allocate host memory
        if (!tensor.isInput) {
            err = cudaHostAlloc(&tensor.hostBuffer, tensor.size, cudaHostAllocDefault);
            if (err != cudaSuccess) {
                RCLCPP_ERROR(this->logger, "CUDA host alloc failed for tensor %s: %s", 
                             tensor.name.c_str(), cudaGetErrorString(err));
                throw std::runtime_error("CUDA host alloc failed");
            }
        }
        
        // Store device buffer for inference
        deviceBuffers.push_back(tensor.deviceBuffer);
    }

    // Warmup the model to reduce first inference latency
    if (warmup) {
        RCLCPP_INFO(this->logger, "Warming up model...");
        for (int i = 0; i < 10; i++) {
            // Create dummy input data
            void* dummyInput = malloc(tensors[inputBlobName].size);
            if (!dummyInput) {
                RCLCPP_ERROR(this->logger, "Failed to allocate memory for warmup");
                throw std::runtime_error("Failed to allocate memory for warmup");
            }
            
            memset(dummyInput, 0, tensors[inputBlobName].size);
            cudaError_t err = cudaMemcpyAsync(
                tensors[inputBlobName].deviceBuffer, 
                dummyInput, 
                tensors[inputBlobName].size, 
                cudaMemcpyHostToDevice, 
                this->stream
            );
            free(dummyInput);
            
            if (err != cudaSuccess) {
                RCLCPP_ERROR(this->logger, "CUDA memcpy failed during warmup: %s", cudaGetErrorString(err));
                throw std::runtime_error("CUDA memcpy failed during warmup");
            }
            
            // Run inference
            try {
                this->infer();
            } catch (const std::exception& e) {
                RCLCPP_ERROR(this->logger, "Inference failed during warmup: %s", e.what());
                throw;
            }
        }
        RCLCPP_INFO(this->logger, "Model warmup completed (10 iterations)");
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
    
    // Get dimensions from tensor
    int height = inputTensor.dims.d[2]; // Assuming NCHW format
    int width = inputTensor.dims.d[3];
    cv::Size size{width, height};
    
    // Preprocess image
    this->letterbox(image, nchw, size);
    
    // Update execution context with input dimensions if dynamic
    if (this->engine->getTensorShape(inputBlobName.c_str()).d[0] == -1) {
        nvinfer1::Dims4 inputDims{1, 3, height, width};
        if (!this->context->setInputShape(inputBlobName.c_str(), inputDims)) {
            RCLCPP_ERROR(this->logger, "Failed to set input shape for dynamic batch");
            throw std::runtime_error("Failed to set input shape");
        }
    }
    
    // Copy to device
    cudaError_t err = cudaMemcpyAsync(
        inputTensor.deviceBuffer,
        nchw.ptr<float>(),
        nchw.total() * nchw.elemSize(),
        cudaMemcpyHostToDevice,
        this->stream
    );
    
    if (err != cudaSuccess) {
        RCLCPP_ERROR(this->logger, "Failed to copy input data to device: %s", cudaGetErrorString(err));
        throw std::runtime_error("CUDA memcpy failed");
    }
}

void YOLOv8::copy_from_Mat(const cv::Mat& image, const cv::Size& size)
{
    auto& inputTensor = tensors[inputBlobName];
    cv::Mat nchw;
    
    // Preprocess image
    this->letterbox(image, nchw, size);
    
    // Update execution context with input dimensions if dynamic
    if (this->engine->getTensorShape(inputBlobName.c_str()).d[0] == -1) {
        nvinfer1::Dims4 inputDims{1, 3, size.height, size.width};
        if (!this->context->setInputShape(inputBlobName.c_str(), inputDims)) {
            RCLCPP_ERROR(this->logger, "Failed to set input shape for dynamic batch");
            throw std::runtime_error("Failed to set input shape");
        }
    }
    
    // Copy to device
    cudaError_t err = cudaMemcpyAsync(
        inputTensor.deviceBuffer,
        nchw.ptr<float>(),
        nchw.total() * nchw.elemSize(),
        cudaMemcpyHostToDevice,
        this->stream
    );
    
    if (err != cudaSuccess) {
        RCLCPP_ERROR(this->logger, "Failed to copy input data to device: %s", cudaGetErrorString(err));        throw std::runtime_error("CUDA memcpy failed");
    }
}

void YOLOv8::infer()
{
    // 1) Set tensor addresses (name-based)
    for (const auto& name : inputNames) {
        if (!this->context->setTensorAddress(name.c_str(), tensors[name].deviceBuffer)) {
            RCLCPP_ERROR(this->logger, "Failed to setTensorAddress for input: %s", name.c_str());
            throw std::runtime_error("setTensorAddress failed (input)");
        }
    }

    for (const auto& name : outputNames) {
        if (!this->context->setTensorAddress(name.c_str(), tensors[name].deviceBuffer)) {
            RCLCPP_ERROR(this->logger, "Failed to setTensorAddress for output: %s", name.c_str());
            throw std::runtime_error("setTensorAddress failed (output)");
        }
    }

    // 2) Enqueue inference (TRT 10+)
    bool status = this->context->enqueueV3(this->stream);
    if (!status) {
        RCLCPP_ERROR(this->logger, "TensorRT inference failed (enqueueV3)");
        throw std::runtime_error("TensorRT inference failed (enqueueV3)");
    }

    // 3) Copy outputs device -> host
    for (const auto& name : outputNames) {
        auto& tensor = tensors[name];
        cudaError_t err = cudaMemcpyAsync(
            tensor.hostBuffer,
            tensor.deviceBuffer,
            tensor.size,
            cudaMemcpyDeviceToHost,
            this->stream
        );

        if (err != cudaSuccess) {
            RCLCPP_ERROR(this->logger, "Failed to copy output %s: %s", name.c_str(), cudaGetErrorString(err));
            throw std::runtime_error("CUDA memcpy failed (output)");
        }
    }

    // 4) Sync
    cudaError_t err = cudaStreamSynchronize(this->stream);
    if (err != cudaSuccess) {
        RCLCPP_ERROR(this->logger, "CUDA stream synchronize failed: %s", cudaGetErrorString(err));
        throw std::runtime_error("CUDA stream synchronize failed");
    }
}


usv_interfaces::msg::ZbboxArray YOLOv8::postprocess()
{
    usv_interfaces::msg::ZbboxArray arr;

    // Access output tensors by name
    int*   num_dets = static_cast<int*>(tensors[numDetectionsName].hostBuffer);
    float* boxes    = static_cast<float*>(tensors[boxesName].hostBuffer);
    float* scores   = static_cast<float*>(tensors[scoresName].hostBuffer);
    int*   labels   = static_cast<int*>(tensors[classesName].hostBuffer);
    
    auto& dw       = this->pparam.dw;
    auto& dh       = this->pparam.dh;
    auto& width    = this->pparam.width;
    auto& height   = this->pparam.height;
    auto& ratio    = this->pparam.ratio;

    RCLCPP_DEBUG(this->logger, "num_dets: %d", num_dets[0]);
    RCLCPP_DEBUG(this->logger, "threshold: %f", this->threshold);

    for (int i = 0; i < num_dets[0]; i++) {
        float* ptr = boxes + i * 4;
        float prob = *(scores + i);

        RCLCPP_DEBUG(this->logger, "score of [%f] with threshold [%f]", prob, this->threshold);
        
        // Apply threshold filtering
        if (prob < this->threshold) {
            continue;
        }

        float x0 = *ptr++ - dw;
        float y0 = *ptr++ - dh;
        float x1 = *ptr++ - dw;
        float y1 = *ptr - dh;

        x0 = clamp(x0 * ratio, 0.f, width);
        y0 = clamp(y0 * ratio, 0.f, height);
        x1 = clamp(x1 * ratio, 0.f, width);
        y1 = clamp(y1 * ratio, 0.f, height);

        usv_interfaces::msg::Zbbox obj;
        obj.uuid = sl::generate_unique_id();
        obj.prob = prob;
        obj.label = *(labels + i);
        
        obj.x0 = x0;
        obj.y0 = y0;
        obj.x1 = x1;
        obj.y1 = y1;

        arr.boxes.push_back(obj);
    }

    RCLCPP_DEBUG(this->logger, "Detected objects: %zu", arr.boxes.size());
    
    return arr;
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


#endif  // JETSON_DETECT_YOLOV8_HPP
