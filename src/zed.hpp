#ifndef ZED_USV
#define ZED_USV
#include "sl/Camera.hpp"
#include "rclcpp/rclcpp.hpp"

class ZED_usv {
public:
    sl::Camera cam;

    ZED_usv(rclcpp::Logger);
    ~ZED_usv();

private:
    sl::InitParameters init_params;
    sl::ObjectDetectionParameters detection_params;

    rclcpp::Logger logger;
};

ZED_usv::ZED_usv(rclcpp::Logger logger_param) : logger(logger_param)
{   
    this->logger = logger_param;

    init_params.sdk_verbose = true; // Consider setting to false in production
    init_params.depth_mode = sl::DEPTH_MODE::ULTRA;
    init_params.coordinate_system = sl::COORDINATE_SYSTEM::LEFT_HANDED_Z_UP;
    init_params.coordinate_units = sl::UNIT::METER;
    init_params.depth_maximum_distance = 20;

    auto returned_state = cam.open(init_params);
    if (returned_state != sl::ERROR_CODE::SUCCESS) {
        RCLCPP_ERROR(this->logger, "Failed to open camera: %s", sl::toString(returned_state).c_str());
        throw std::runtime_error("Failed to open ZED camera");
    }

    auto tracking_state = cam.enablePositionalTracking();
    if (tracking_state != sl::ERROR_CODE::SUCCESS) {
        RCLCPP_WARN(this->logger, "Failed to enable positional tracking: %s", 
                   sl::toString(tracking_state).c_str());
        // Continue anyway as this might not be critical
    }
    
    detection_params.enable_tracking = false; // Consider setting to true for tracking
    detection_params.enable_segmentation = false;
    detection_params.detection_model = sl::OBJECT_DETECTION_MODEL::CUSTOM_BOX_OBJECTS;

    returned_state = cam.enableObjectDetection(detection_params);
    if (returned_state != sl::ERROR_CODE::SUCCESS) {
        RCLCPP_ERROR(this->logger, "Failed to enable object detection in ZED SDK: %s", 
                    sl::toString(returned_state).c_str());
        cam.close();
        throw std::runtime_error("Failed to enable object detection in ZED SDK");
    }

    RCLCPP_INFO(this->logger, "ZED camera initialized successfully");
}

ZED_usv::~ZED_usv()
{
    RCLCPP_DEBUG(this->logger, "se ha cerrado la camara");
    cam.close();
}

#endif
