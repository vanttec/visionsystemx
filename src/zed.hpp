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

    // OPTIMIZED SETTINGS FOR JETSON ORIN NX AND NANO - USB STABILITY FIX

    init_params.sdk_verbose = true;  // ORIGINAL
    // init_params.sdk_verbose = false;    // OPTIMIZED
    

    init_params.depth_mode = sl::DEPTH_MODE::ULTRA;        // ORIGINAL
    // init_params.depth_mode = sl::DEPTH_MODE::NEURAL_LIGHT;   // OPTIMIZED
    
    init_params.coordinate_system = sl::COORDINATE_SYSTEM::LEFT_HANDED_Z_UP;
    init_params.coordinate_units = sl::UNIT::METER;
    init_params.depth_maximum_distance = 20;
    
    init_params.camera_resolution = sl::RESOLUTION::HD1080;  // DEFAULT (not set before)
    init_params.camera_fps = 30;                              // DEFAULT (not set before)
    // init_params.camera_resolution = sl::RESOLUTION::HD720;      // OPTIMIZED
    // init_params.camera_fps = 15;                                 // OPTIMIZED
       
    // init_params.enable_image_enhancement = true;  // DEFAULT
    init_params.enable_image_enhancement = false;    // OPTIMIZED

    // init_params.sensors_required = true;   // DEFAULT
    init_params.sensors_required = false;     // OPTIMIZED

    auto returned_state = cam.open(init_params);
    if (returned_state != sl::ERROR_CODE::SUCCESS) {
        RCLCPP_ERROR(this->logger, "Failed to open camera: %s", sl::toString(returned_state).c_str());
        throw std::runtime_error("Failed to open ZED camera");
    }

    // [ADDED] Log camera info for debugging
    auto info = cam.getCameraInformation();
    RCLCPP_INFO(this->logger, "ZED Camera: Model=%d, S/N=%d, FW=%d", 
                static_cast<int>(info.camera_model), 
                info.serial_number, 
                info.camera_configuration.firmware_version);

    sl::PositionalTrackingParameters tracking_params;
    tracking_params.enable_area_memory = true;     // DEFAULT - saves/loads area maps
    // tracking_params.enable_area_memory = false;    // OPTIMIZED - reduces memory usage
    tracking_params.enable_pose_smoothing = true;     // Keep smoothing for better tracking
    tracking_params.enable_imu_fusion = true;         // Keep IMU fusion for accuracy
    
    // ORIGINAL: cam.enablePositionalTracking();      // Used default parameters
    auto tracking_state = cam.enablePositionalTracking(tracking_params);  // OPTIMIZED
    if (tracking_state != sl::ERROR_CODE::SUCCESS) {
        RCLCPP_WARN(this->logger, "Failed to enable positional tracking: %s", 
                   sl::toString(tracking_state).c_str());
        // Continue anyway as this might not be critical
    }
    
    detection_params.enable_tracking = false;
    detection_params.enable_segmentation = false;
    detection_params.detection_model = sl::OBJECT_DETECTION_MODEL::CUSTOM_BOX_OBJECTS;

    returned_state = cam.enableObjectDetection(detection_params);
    if (returned_state != sl::ERROR_CODE::SUCCESS) {
        RCLCPP_ERROR(this->logger, "Failed to enable object detection in ZED SDK: %s", 
                    sl::toString(returned_state).c_str());
        cam.close();
        throw std::runtime_error("Failed to enable object detection in ZED SDK");
    }

    RCLCPP_INFO(this->logger, "ZED camera initialized successfully (NEURAL_LIGHT, HD720@15fps)");
}

ZED_usv::~ZED_usv()
{
    RCLCPP_DEBUG(this->logger, "se ha cerrado la camara");
    cam.close();
}

#endif
