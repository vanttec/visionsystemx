#ifndef DETECT_NORMAL_COMMON_HPP
#define DETECT_NORMAL_COMMON_HPP
#include "NvInfer.h"
#include "opencv2/opencv.hpp"
#include <sys/stat.h>
#include <unistd.h>

// inline cv::Mat slMat2cvMat(sl::Mat& input) {
//     // Mapping between MAT_TYPE and CV_TYPE
//     int cv_type = -1;
//     switch (input.getDataType()) {
//         case sl::MAT_TYPE::F32_C1: cv_type = CV_32FC1;
//             break;
//         case sl::MAT_TYPE::F32_C2: cv_type = CV_32FC2;
//             break;
//         case sl::MAT_TYPE::F32_C3: cv_type = CV_32FC3;
//             break;
//         case sl::MAT_TYPE::F32_C4: cv_type = CV_32FC4;
//             break;
//         case sl::MAT_TYPE::U8_C1: cv_type = CV_8UC1;
//             break;
//         case sl::MAT_TYPE::U8_C2: cv_type = CV_8UC2;
//             break;
//         case sl::MAT_TYPE::U8_C3: cv_type = CV_8UC3;
//             break;
//         case sl::MAT_TYPE::U8_C4: cv_type = CV_8UC4;
//             break;
//         default: break;
//     }
//
//     return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(sl::MEM::CPU));
// }

#define CHECK(call)                                                                                                    \
    do {                                                                                                               \
        const cudaError_t error_code = call;                                                                           \
        if (error_code != cudaSuccess) {                                                                               \
            printf("CUDA Error:\n");                                                                                   \
            printf("    File:       %s\n", __FILE__);                                                                  \
            printf("    Line:       %d\n", __LINE__);                                                                  \
            printf("    Error code: %d\n", error_code);                                                                \
            printf("    Error text: %s\n", cudaGetErrorString(error_code));                                            \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

// TensorRT logger implementation
class Logger : public nvinfer1::ILogger {
public:
    explicit Logger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kERROR) 
        : severity_(severity) {}
    
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
        if (severity <= severity_) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
private:
    nvinfer1::ILogger::Severity severity_;
};

inline int get_size_by_dims(const nvinfer1::Dims& dims)
{
    int size = 1;
    for (int i = 0; i < dims.nbDims; i++) {
        size *= dims.d[i];
    }
    return size;
}

inline int type_to_size(const nvinfer1::DataType& dataType)
{
    switch (dataType) {
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kINT32:
            return 4;
        case nvinfer1::DataType::kINT8:
            return 1;
        case nvinfer1::DataType::kBOOL:
            return 1;
        default:
            return 4;
    }
}

inline static float clamp(float val, float min, float max)
{
    return val > min ? (val < max ? val : max) : min;
}

inline bool IsPathExist(const std::string& path)
{
    if (access(path.c_str(), 0) == F_OK) {
        return true;
    }
    return false;
}

inline bool IsFile(const std::string& path)
{
    if (!IsPathExist(path)) {
        printf("%s:%d %s not exist\n", __FILE__, __LINE__, path.c_str());
        return false;
    }
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode));
}

inline bool IsFolder(const std::string& path)
{
    if (!IsPathExist(path)) {
        return false;
    }
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0 && S_ISDIR(buffer.st_mode));
}

namespace det {
struct Binding {
    size_t         size  = 1;
    size_t         dsize = 1;
    nvinfer1::Dims dims;
    std::string    name;
};

struct Object {
    cv::Rect_<float> rect;
    int              label = 0;
    float            prob  = 0.0;
};

struct PreParam {
    float ratio  = 1.0f;
    float dw     = 0.0f;
    float dh     = 0.0f;
    float height = 0;
    float width  = 0;
};
}  // namespace det
#endif  // DETECT_NORMAL_COMMON_HPP
