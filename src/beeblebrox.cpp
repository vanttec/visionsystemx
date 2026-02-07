#define GPU
#define USE_ZED_SDK

// ros
#include "cv_bridge/cv_bridge.h"
#include "rclcpp/rclcpp.hpp"    
#include "ament_index_cpp/get_package_share_directory.hpp"
#include "message_filters/subscriber.h"
#include "image_transport/image_transport.hpp"

// msgs
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/distortion_models.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "usv_interfaces/msg/object.hpp"
#include "usv_interfaces/msg/object_list.hpp"
#include "usv_interfaces/msg/zbbox.hpp"
#include "usv_interfaces/msg/zbbox_array.hpp"

// zed sdk
#ifdef USE_ZED_SDK
#include "sl/Camera.hpp"
#endif
#include "sl/Camera.hpp"

// opencv
#include "opencv2/opencv.hpp"

// PCL for pointcloud processing in simulation
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

// cuda
#include <cuda.h>
#include <cuda_runtime_api.h>

// local headers
#include "bytetrack.hpp"
#ifdef USE_ZED_SDK
#include "zed.hpp"
#endif

using std::placeholders::_1;

#ifdef USE_ZED_SDK
// helper function to convert sl::Mat to cv::Mat (from ZED SDK)
inline cv::Mat slMat2cvMat(sl::Mat& input) {
    // mapping between MAT_TYPE and CV_TYPE
    int cv_type = -1;
    switch (input.getDataType()) {
        case sl::MAT_TYPE::F32_C1: cv_type = CV_32FC1;
            break;
        case sl::MAT_TYPE::F32_C2: cv_type = CV_32FC2;
            break;
        case sl::MAT_TYPE::F32_C3: cv_type = CV_32FC3;
            break;
        case sl::MAT_TYPE::F32_C4: cv_type = CV_32FC4;
            break;
        case sl::MAT_TYPE::U8_C1: cv_type = CV_8UC1;
            break;
        case sl::MAT_TYPE::U8_C2: cv_type = CV_8UC2;
            break;
        case sl::MAT_TYPE::U8_C3: cv_type = CV_8UC3;
            break;
        case sl::MAT_TYPE::U8_C4: cv_type = CV_8UC4;
            break;
        default: break;
    }

    return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(sl::MEM::CPU));
}
#endif

// PCL point type for simulation
typedef pcl::PointXYZRGB PointT;

/***
 * class for interfacing with the multiple detectors and syncronizing their results
***/
class DetectorInterface: public rclcpp::Node {

private:
    cv::Mat         res, image;
    cv::Size        size        = cv::Size{640, 640};
    int             num_labels  = 80;
    int             topk        = 100;
    float           score_thres = 0.25f;
    float           iou_thres   = 0.65f;
    
    // flag to determine if we're in simulation mode
    bool            simulation_mode = false;

#ifdef USE_ZED_SDK
    sl::ObjectDetectionRuntimeParameters    object_tracker_parameters_rt;
#endif

    std::string     engine_path;
    std::string     classes_path;
    std::string     output_topic;

    // final detections publishers
    rclcpp::Publisher<usv_interfaces::msg::ObjectList>::SharedPtr yolo_pub;
    rclcpp::Publisher<usv_interfaces::msg::ObjectList>::SharedPtr shapes_pub;

    // image_transport publisher
    image_transport::Publisher ip;
    image_transport::Publisher depth_image_publisher;

    // publish camera info
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_pub;

    // receiver of boundingboxes
    rclcpp::Subscription<usv_interfaces::msg::ZbboxArray>::SharedPtr yolo_sub;
    rclcpp::Subscription<usv_interfaces::msg::ZbboxArray>::SharedPtr shapes_sub;
    
    // pointcloud subscriber for simulation
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub;

    // timer in which to execute everything
    rclcpp::TimerBase::SharedPtr timer;

    // dynamic parameters
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;
    double high_threshold = 0.6;
    double low_threshold = 0.1;
    int max_time_lost = 30;
    int min_hits = 3;

#ifdef USE_ZED_SDK
    sl::Mat left_sl, point_cloud;
    // instance of the zed camera
    ZED_usv zed_interface;
#endif
    
    // store the latest image and depth for simulation
    cv::Mat latest_image;
    cv::Mat latest_depth;
    
    // dimensions of the latest pointcloud
    int cloud_width = 0;
    int cloud_height = 0;

    // tracker
    bytetrack::ByteTrack yolo_tracker;

    // optional: camera parameter
    // (only needed for unorganized pointclouds)
    double fx = 500.0;
    double fy = 500.0;
    double cx = 320.0;
    double cy = 240.0;
    int img_width = 640;
    int img_height = 480;

/**
 * extract data from an organized pointcloud
 * simplified approach on gz.py
 * @param cloud_msg: input pointcloud message
 */
void process_organized_pointcloud(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud_msg) {
    // check if the cloud is organized
    if (cloud_msg->height <= 1) {
        RCLCPP_ERROR(this->get_logger(), "Received an unorganized pointcloud. Cannot process as organized.");
        return;
    }
    
    cloud_width = cloud_msg->width;
    cloud_height = cloud_msg->height;

	if (cloud_width <= 0 || cloud_height <= 0) {
    	RCLCPP_ERROR(this->get_logger(), "Invalid pointcloud dimensions: %dx%d", 
                 cloud_width, cloud_height);
    	return;
	}
    
    // create empty RGB and depth images
    cv::Mat rgb_img(cloud_height, cloud_width, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat depth_img(cloud_height, cloud_width, CV_32FC1, cv::Scalar(0.0f));
    
    // get field offsets for x, y, z, rgb
    int x_offset = -1, y_offset = -1, z_offset = -1, rgb_offset = -1;
    bool has_rgb = false;
    
    for (const auto& field : cloud_msg->fields) {
        if (field.name == "x") {
            x_offset = field.offset;
        } else if (field.name == "y") {
            y_offset = field.offset;
        } else if (field.name == "z") {
            z_offset = field.offset;
        } else if (field.name == "rgb" || field.name == "rgba") {
            rgb_offset = field.offset;
            has_rgb = true;
        }
    }
    
    // check if we have all the required fields
    if (x_offset < 0 || y_offset < 0 || z_offset < 0) {
        RCLCPP_ERROR(this->get_logger(), "Pointcloud missing required fields (x, y, z)");
        return;
    }
    
    // extract data from the pointcloud
    for (uint32_t row = 0; row < cloud_msg->height; ++row) {
        for (uint32_t col = 0; col < cloud_msg->width; ++col) {
            // calculate index in the raw data
            size_t point_idx = row * cloud_msg->row_step + col * cloud_msg->point_step;
            
            // extract x, y, z values (depth)
            float x = 0.0f, y = 0.0f, z = 0.0f;
            memcpy(&x, &cloud_msg->data[point_idx + x_offset], sizeof(float));
            memcpy(&y, &cloud_msg->data[point_idx + y_offset], sizeof(float));
            memcpy(&z, &cloud_msg->data[point_idx + z_offset], sizeof(float));
            
            // store depth (z value) in the depth image
            if (!std::isfinite(z)) {
                depth_img.at<float>(row, col) = 0.0f;  // Mark as invalid
            } else {
                depth_img.at<float>(row, col) = z;
            }
            
            // extract RGB if available
            if (has_rgb) {
                uint32_t rgb_val = 0;
                memcpy(&rgb_val, &cloud_msg->data[point_idx + rgb_offset], sizeof(uint32_t));
                
                // extract RGB components
                uint8_t r = (rgb_val >> 16) & 0xFF;
                uint8_t g = (rgb_val >> 8) & 0xFF;
                uint8_t b = rgb_val & 0xFF;
                
                rgb_img.at<cv::Vec3b>(row, col) = cv::Vec3b(r, g, b);
            }
        }
    }
    
    // fill small holes in the RGB image (optional)
    if (has_rgb) {
        cv::medianBlur(rgb_img, rgb_img, 3);
    }
    
    // store the results
    latest_image = rgb_img;
    latest_depth = depth_img;
    
    // set image dimensions from the pointcloud
    img_width = cloud_width;
    img_height = cloud_height;
    
    // publish RGB image
    std_msgs::msg::Header hdr = cloud_msg->header;
	cv::cvtColor(rgb_img, rgb_img, cv::COLOR_RGB2BGR);
	sensor_msgs::msg::Image::SharedPtr rgb_msg = cv_bridge::CvImage(hdr, "bgr8", rgb_img).toImageMsg();
    this->ip.publish(rgb_msg);
    
    // publish depth visualization
    cv::Mat depth_norm;
    cv::Mat depth_viz = depth_img.clone();
    // replace NaN/Inf with 0
    for (int i = 0; i < depth_viz.rows; ++i) {
        for (int j = 0; j < depth_viz.cols; ++j) {
            if (!std::isfinite(depth_viz.at<float>(i, j))) {
                depth_viz.at<float>(i, j) = 0.0f;
            }
        }
    }
    // normalize and colorize depth
    cv::normalize(depth_viz, depth_norm, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::Mat depth_color;
    cv::applyColorMap(depth_norm, depth_color, cv::COLORMAP_JET);
    
    sensor_msgs::msg::Image::SharedPtr depth_msg = 
        cv_bridge::CvImage(hdr, "bgr8", depth_color).toImageMsg();
    this->depth_image_publisher.publish(depth_msg);
    
    // publish camera info
    sensor_msgs::msg::CameraInfo camInfoMsg;
    camInfoMsg.distortion_model = sensor_msgs::distortion_models::PLUMB_BOB;
    
    camInfoMsg.d.resize(5, 0.0);  // no distortion
    
    // we don't know the actual camera parameters from the organized pointcloud
    // so we'll just set reasonable defaults based on the image dimensions
    camInfoMsg.k.fill(0.0);
    camInfoMsg.k[0] = cloud_width / 2.0;  // Approximate fx
    camInfoMsg.k[2] = cloud_width / 2.0;  // Approximate cx
    camInfoMsg.k[4] = cloud_height / 2.0; // Approximate fy
    camInfoMsg.k[5] = cloud_height / 2.0; // Approximate cy
    camInfoMsg.k[8] = 1.0;
    
    camInfoMsg.r.fill(0.0);
    for (size_t i = 0; i < 3; i++) {
        camInfoMsg.r[i + i * 3] = 1;  // identity matrix
    }
    
    camInfoMsg.p.fill(0.0);
    camInfoMsg.p[0] = camInfoMsg.k[0];
    camInfoMsg.p[2] = camInfoMsg.k[2];
    camInfoMsg.p[5] = camInfoMsg.k[4];
    camInfoMsg.p[6] = camInfoMsg.k[5];
    camInfoMsg.p[10] = 1.0;
    
    camInfoMsg.width = cloud_width;
    camInfoMsg.height = cloud_height;
    camInfoMsg.header = hdr;
    
    this->cam_info_pub->publish(camInfoMsg);
}

/***
 * callback for pointcloud in simulation mode
 * @param cloud_msg: input pointcloud
***/
void receive_pointcloud(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg) {
    RCLCPP_DEBUG(this->get_logger(), "[simulation] received pointcloud %dx%d", 
                 cloud_msg->width, cloud_msg->height);
    
    // check if the pointcloud is organized
    if (cloud_msg->height > 1) {
        // process as organized pointcloud
        process_organized_pointcloud(cloud_msg);
    } else {
        RCLCPP_WARN(this->get_logger(), "Received unorganized pointcloud - this mode is not supported for simulation. Please use organized pointclouds.");
    }
}

/***
 * send frame to ros (real-life mode with ZED SDK)
***/
void frame_send() {
#ifdef USE_ZED_SDK
    if (!simulation_mode) {
        if (zed_interface.cam.grab() == sl::ERROR_CODE::SUCCESS) {
            
            /* publish zed image */

            // retrieve image from zed
            zed_interface.cam.retrieveImage(left_sl, sl::VIEW::LEFT);

            // convert to opencv format
            cv::Mat img = slMat2cvMat(left_sl);
            cv::cvtColor(img, img, cv::COLOR_BGRA2BGR);

            // convert to ros2 message
            std_msgs::msg::Header hdr;
            sensor_msgs::msg::Image::SharedPtr msg = cv_bridge::CvImage(hdr, "bgr8", img).toImageMsg();
            msg->header.stamp = this->get_clock()->now();

            // publish
            this->ip.publish(msg);
            
            sensor_msgs::msg::CameraInfo leftCamInfoMsg;

            /* publish camera info parameters */

            sl::CalibrationParameters zedParam;
            zedParam = zed_interface.cam.getCameraInformation().camera_configuration.calibration_parameters;
            
            sl::Resolution res;
            res = zed_interface.cam.getCameraInformation().camera_configuration.resolution;

            // only for ZED2, ZED2i, ZEDX and ZEDXm
            leftCamInfoMsg.distortion_model = sensor_msgs::distortion_models::RATIONAL_POLYNOMIAL;

            // yes
            leftCamInfoMsg.d.resize(8);
            leftCamInfoMsg.d[0] = zedParam.left_cam.disto[0];    // k1
            leftCamInfoMsg.d[1] = zedParam.left_cam.disto[1];    // k2
            leftCamInfoMsg.d[2] = zedParam.left_cam.disto[2];    // p1
            leftCamInfoMsg.d[3] = zedParam.left_cam.disto[3];    // p2
            leftCamInfoMsg.d[4] = zedParam.left_cam.disto[4];    // k3
            leftCamInfoMsg.d[5] = zedParam.left_cam.disto[5];    // k4
            leftCamInfoMsg.d[6] = zedParam.left_cam.disto[6];    // k5
            leftCamInfoMsg.d[7] = zedParam.left_cam.disto[7];    // k6

            // yes
            leftCamInfoMsg.k.fill(0.0);
            leftCamInfoMsg.k[0] = static_cast<double>(zedParam.left_cam.fx);
            leftCamInfoMsg.k[2] = static_cast<double>(zedParam.left_cam.cx);
            leftCamInfoMsg.k[4] = static_cast<double>(zedParam.left_cam.fy);
            leftCamInfoMsg.k[5] = static_cast<double>(zedParam.left_cam.cy);
            leftCamInfoMsg.k[8] = 1.0;

            // yes
            leftCamInfoMsg.r.fill(0.0);
            for (size_t i = 0; i < 3; i++) {
                // identity
                leftCamInfoMsg.r[i + i * 3] = 1;
            }

            // yes
            leftCamInfoMsg.p.fill(0.0);
            leftCamInfoMsg.p[0] = static_cast<double>(zedParam.left_cam.fx);
            leftCamInfoMsg.p[2] = static_cast<double>(zedParam.left_cam.cx);
            leftCamInfoMsg.p[5] = static_cast<double>(zedParam.left_cam.fy);
            leftCamInfoMsg.p[6] = static_cast<double>(zedParam.left_cam.cy);
            leftCamInfoMsg.p[10] = 1.0;

            // yes
            leftCamInfoMsg.width = static_cast<uint32_t>(res.width);
            leftCamInfoMsg.height = static_cast<uint32_t>(res.height);
            leftCamInfoMsg.header.frame_id = "zed2i_left_camera_optical_frame";
            
            this->cam_info_pub->publish(leftCamInfoMsg);
        }
    }
#endif
    // in simulation mode, the frame and camera info are sent in the pointcloud callback
}

/***
 * get 3D position from simulation (organized pointcloud)
 * @param bbox: 2D bbox
 * @return 3D position
***/
std::array<float, 3> get_position_from_simulation(const usv_interfaces::msg::Zbbox& bbox) {
    if (latest_depth.empty()) {
        return {0.0f, 0.0f, 0.0f};
    }
    
    // get center point of bounding box
    int center_x = (bbox.x0 + bbox.x1) / 2;
    int center_y = (bbox.y0 + bbox.y1) / 2;
    
    // ensure within bounds
    center_x = std::max(0, std::min(center_x, latest_depth.cols - 1));
    center_y = std::max(0, std::min(center_y, latest_depth.rows - 1));
    
    // direct access to depth from the organized pointcloud
    float depth = latest_depth.at<float>(center_y, center_x);
    
    if (depth <= 0 || std::isnan(depth)) {
        // if center has no valid depth, try to find a valid depth point within the bounding box
        float sum_depth = 0.0f;
        int valid_points = 0;
        
        for (int y = bbox.y0; y <= bbox.y1; y++) {
            for (int x = bbox.x0; x <= bbox.x1; x++) {
                if (x >= 0 && x < latest_depth.cols && y >= 0 && y < latest_depth.rows) {
                    float d = latest_depth.at<float>(y, x);
                    if (d > 0 && !std::isnan(d)) {
                        sum_depth += d;
                        valid_points++;
                    }
                }
            }
        }
        
        if (valid_points > 0) {
            depth = sum_depth / valid_points;
        } else {
            return {0.0f, 0.0f, 0.0f};  // no valid depth found
        }
    }
    
    // for organized pointclouds, x and y can be directly derived from the depth and pixel coordinates
    // we're using the pointcloud's natural coordinate system
    // these calculations assume the pointcloud's coordinate system
    float z = depth;
    float x = ((center_x - latest_depth.cols / 2.0f) * z) / (latest_depth.cols / 2.0f);
    float y = ((center_y - latest_depth.rows / 2.0f) * z) / (latest_depth.rows / 2.0f);
    
    return {x, y, z};
}

/***
 * callback for detections (YOLO)
 * @param dets: yolo detections
***/
void receive_yolo(const usv_interfaces::msg::ZbboxArray::SharedPtr dets) {
    if (!dets || dets->boxes.empty()) {
        RCLCPP_DEBUG(this->get_logger(), "[yolo] received empty detection message");
        return;
    }

    RCLCPP_DEBUG(this->get_logger(), "[yolo] received detections: %ld", dets->boxes.size());
    
    // Apply ByteTrack to incoming detections
    // usv_interfaces::msg::ZbboxArray tracked_dets = yolo_tracker.update(*dets);
    // RCLCPP_DEBUG(this->get_logger(), "[yolo] tracked detections: %ld", tracked_dets.boxes.size());
    
    if (simulation_mode) {
        // process detections in simulation mode
        usv_interfaces::msg::ObjectList detections;

        for (const auto& det : dets->boxes) {
            auto position = get_position_from_simulation(det);
            
            // RED
            // GREEN
            // BLUE
            // YELLOW
            // BLACK

            RCLCPP_INFO(this->get_logger(), "[yolo] debug label: %ld", det.label);

            usv_interfaces::msg::Object obj;
            switch (det.label) {
                case 0:  // black
                    obj.color = 4;
                    obj.type = "round";
                    break;
                case 1:  // blue
                    obj.color = 2;
                    obj.type = "round";
                    break;
                case 2: // course marker
                    obj.color = 4;
                    obj.type = "marker";
                    break;
                case 3: // green
                    obj.color = 1;
                    obj.type = "round";
                    break;
                case 4: // red marker
                    obj.color = 0;
                    obj.type = "marker";
                    break;
                case 5: // red
                    obj.color = 0;
                    obj.type = "round";
                    break;
                case 6: // green marker
                    obj.color = 1;
                    obj.type = "marker";
                    break;
                case 7: // yellow
                    obj.color = 3;
                    obj.type = "round";
                    break;
                default:
                    obj.color = -1;
                    obj.type = "ignore";
                    break;
            }
            
            obj.x = position[0];
            obj.y = position[1];
            
            if (std::isnan(obj.x) || std::isnan(obj.y)) {
                obj.x = 0.0;
                obj.y = 0.0;
                obj.type = "ignore";
                RCLCPP_DEBUG(this->get_logger(), "[yolo] NaN on coordinates");
            }
            
            // Preserve the track ID if available
            if (!det.uuid.empty()) {
                obj.uuid = det.uuid;
            }
            
            detections.obj_list.push_back(obj);
        }
        
        this->yolo_pub->publish(detections);
    }
#ifdef USE_ZED_SDK
    else {
        // real-life mode using ZED SDK
        std::vector<sl::CustomBoxObjectData> dets_sl;
        
        // convert from ros message to global format
        to_sl(dets_sl, dets);
        // to_sl(dets_sl, std::make_shared<usv_interfaces::msg::ZbboxArray>(dets));

        if (dets_sl.empty()) {
            RCLCPP_DEBUG(this->get_logger(), "[yolo] no valid detections to process");
            return;
        }

        auto start = std::chrono::system_clock::now();

        try {
            // send to zed sdk
            sl::Objects out_objs;
            auto result = zed_interface.cam.ingestCustomBoxObjects(dets_sl);
            if (result != sl::ERROR_CODE::SUCCESS) {
                RCLCPP_WARN(this->get_logger(), "[yolo] Failed to ingest box objects: %s", 
                            sl::toString(result).c_str());
                return;
            }
            
            result = zed_interface.cam.retrieveObjects(out_objs, object_tracker_parameters_rt);
            if (result != sl::ERROR_CODE::SUCCESS) {
                RCLCPP_WARN(this->get_logger(), "[yolo] Failed to retrieve objects: %s", 
                            sl::toString(result).c_str());
                return;
            }

            auto end = std::chrono::system_clock::now();
            auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
            
            // publish detections in pointcloud
            usv_interfaces::msg::ObjectList detections = objs2markers(out_objs);
            
            // Add the tracking UUIDs to the output object list
            // for (size_t i = 0; i < detections.obj_list.size() && i < tracked_dets.boxes.size(); ++i) {
            //     if (!tracked_dets.boxes[i].uuid.empty()) {
            //         detections.obj_list[i].uuid = tracked_dets.boxes[i].uuid;
            //     }
            // }
            
            this->yolo_pub->publish(detections);

            RCLCPP_DEBUG(this->get_logger(), "[yolo] pointcloud estimation done: %2.4lf ms [%ld]", 
                        tc, detections.obj_list.size());
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "[yolo] error processing detections: %s", e.what());
        }
    }
#endif
}

/***
 * callback for detections (shapes)
 * @param dets: shapes detections
***/
void receive_shapes(const usv_interfaces::msg::ZbboxArray::SharedPtr dets) {
    if (!dets || dets->boxes.empty()) {
        RCLCPP_DEBUG(this->get_logger(), "[shapes] received empty detection message");
        return;
    }
    
    RCLCPP_DEBUG(this->get_logger(), "[shapes] received detections: %ld", dets->boxes.size());
    
    if (simulation_mode) {
        // process detections in simulation mode
        usv_interfaces::msg::ObjectList detections;
        
        for (const auto& det : dets->boxes) {
            auto position = get_position_from_simulation(det);
            
            // RED
            // GREEN
            // BLUE
            // YELLOW
            // BLACK

            usv_interfaces::msg::Object obj;
            switch (det.label) {
                case 0:     // blue circle
                    obj.color = 2;
                    obj.type = "circle";
                    break;
                case 1:     // blue plus
                    obj.color = 2;
                    obj.type = "plus";
                    break;
                case 2:     // blue square
                    obj.color = 2;
                    obj.type = "square";
                    break;
                case 3:     // blue triangle
                    obj.color = 2;
                    obj.type = "triangle";
                    break;
                case 4:     // duck
                    obj.color = 3;
                    obj.type = "duck";
                    break;
                case 5:     // green circle
                    obj.color = 1;
                    obj.type = "circle";
                    break;
                case 6:     // green plus
                    obj.color = 1;
                    obj.type = "plus";
                    break;
                case 7:     // green square
                    obj.color = 1;
                    obj.type = "square";
                    break;
                case 8:     // green triangle
                    obj.color = 1;
                    obj.type = "triangle";
                    break;
                case 9:     // red circle
                    obj.color = 0;
                    obj.type = "circle";
                    break;
                case 10:    // red plus
                    obj.color = 0;
                    obj.type = "plus";
                    break;
                case 11:    // red square
                    obj.color = 0;
                    obj.type = "square";
                    break;
                case 12:    // red triangle
                    obj.color = 0;
                    obj.type = "triangle";
                    break;
                default:
                    obj.color = -1;
                    obj.type = "ignore";
                    break;
            }
            
            obj.x = position[0];
            obj.y = position[1];
            
            if (std::isnan(obj.x) || std::isnan(obj.y)) {
                obj.x = 0.0;
                obj.y = 0.0;
                obj.type = "ignore";
            }
            
            detections.obj_list.push_back(obj);
        }
        
        this->shapes_pub->publish(detections);
    }
#ifdef USE_ZED_SDK
    else {
        // real-life mode using ZED SDK
        std::vector<sl::CustomBoxObjectData> dets_sl;
        
        // convert from ros message to global format
        to_sl(dets_sl, dets);

        auto start = std::chrono::system_clock::now();

        // send to zed sdk
        sl::Objects out_objs;
        zed_interface.cam.ingestCustomBoxObjects(dets_sl);
        zed_interface.cam.retrieveObjects(out_objs, object_tracker_parameters_rt);

        auto end = std::chrono::system_clock::now();
        auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
        
        // publish detections in pointcloud
        usv_interfaces::msg::ObjectList detections = objs2markers_shapes(out_objs);
        this->shapes_pub->publish(detections);

        RCLCPP_DEBUG(this->get_logger(), "[shapes] pointcloud estimation done: %2.4lf ms [%ld]", tc, detections.obj_list.size());
    }
#endif
}

#ifdef USE_ZED_SDK
/***
 * convert from ros message to global format for ZED SDK
 * @param sl_dets: global format
 * @param dets: ros message
***/
void to_sl(std::vector<sl::CustomBoxObjectData>& sl_dets, const usv_interfaces::msg::ZbboxArray::SharedPtr dets) {
    for (auto& det : dets->boxes) {
        sl::CustomBoxObjectData sl_det;
        sl_det.label = det.label;
        sl_det.probability = det.prob;
        
        // Use the UUID from ByteTrack if available, otherwise use the default UUID
        if (!det.uuid.empty()) {
            sl_det.unique_object_id = sl::String(det.uuid.data());
        } else {
            // If there's no UUID in the detection, use the one from ZED SDK
            sl_det.unique_object_id = sl::String(det.uuid.data());
        }

        std::vector<sl::uint2> bbox(4);
        bbox[0] = sl::uint2(det.x0, det.y0);
        bbox[1] = sl::uint2(det.x1, det.y0);
        bbox[2] = sl::uint2(det.x1, det.y1);
        bbox[3] = sl::uint2(det.x0, det.y1);

        sl_det.bounding_box_2d = bbox;
        sl_det.is_grounded = false; // si es `true`, la zed no rastreara este objeto

        sl_dets.push_back(sl_det);
    }
}
#endif

/***
 * convert from global format to message understandable by usv_control
 * @param objs: global format
 * @return ros message
***/
usv_interfaces::msg::ObjectList objs2markers_shapes(sl::Objects objs) {
    usv_interfaces::msg::ObjectList ma;

    for (int i = 0; i < 10; i++) {
        try {
            auto& obj = objs.object_list.at(i);

            usv_interfaces::msg::Object o;

            int color = 0;

            switch (obj.raw_label) {
                case 0:     // blue circle
                    o.color = 2;
                    o.type = "circle";
                    break;
                case 1:     // blue plus
                    o.color = 2;
                    o.type = "plus";
                    break;
                case 2:     // blue square
                    o.color = 2;
                    o.type = "square";
                    break;
                case 3:     // blue triangle
                    o.color = 2;
                    o.type = "triangle";
                    break;
                case 4:     // duck
                    o.color = 3;
                    o.type = "duck";
                    break;
                case 5:     // green circle
                    o.color = 1;
                    o.type = "circle";
                    break;
                case 6:     // green plus
                    o.color = 1;
                    o.type = "plus";
                    break;
                case 7:     // green square
                    o.color = 1;
                    o.type = "square";
                    break;
                case 8:     // green triangle
                    o.color = 1;
                    o.type = "triangle";
                    break;
                case 9:     // red circle
                    o.color = 0;
                    o.type = "circle";
                    break;
                case 10:    // red plus
                    o.color = 0;
                    o.type = "plus";
                    break;
                case 11:    // red square
                    o.color = 0;
                    o.type = "square";
                    break;
                case 12:    // red triangle
                    o.color = 0;
                    o.type = "triangle";
                    break;
                default: // TODO
                    color = -1;
                    o.type = "ignore";
                    break;
            }

            o.x = obj.position[0];
            if ( std::isnan(o.x) ) {
                o.x = 0.0;
            }

            o.y = obj.position[1];
            if ( std::isnan(o.y) ) {
                o.y = 0.0;
            }

            ma.obj_list.push_back(o);
        }
        catch (const std::out_of_range& oor) {
            usv_interfaces::msg::Object o;

            o.color = -1;
            o.x = 0;
            o.y = 0;
            o.type = "ignore";
            ma.obj_list.push_back(o);
        }
    }
    
    return ma;
}

/***
 * convert from global format to message understandable by usv_control
 * @param objs: global format
 * @return ros message
***/
usv_interfaces::msg::ObjectList objs2markers(sl::Objects objs) {
    usv_interfaces::msg::ObjectList ma;

    for (int i = 0; i < 10; i++) {
        try {
            auto& obj = objs.object_list.at(i);

            usv_interfaces::msg::Object o;

            // Set object UUID from ZED tracking ID
            // This will be overwritten later with ByteTrack UUID if available
            o.uuid = std::to_string(obj.id);

            switch (obj.raw_label) {
                case 0:  // black
                    o.color = 4;
                    o.type = "round";
                    break;
                case 1:  // blue
                    o.color = 2;
                    o.type = "round";
                    break;
                case 2: // course marker
                    o.color = 4;
                    o.type = "marker";
                    break;
                case 3: // green
                    o.color = 1;
                    o.type = "round";
                    break;
                case 4: // red marker
                    o.color = 0;
                    o.type = "marker";
                    break;
                case 5: // red
                    o.color = 0;
                    o.type = "round";
                    break;
                case 6: // green marker
                    o.color = 1;
                    o.type = "marker";
                    break;
                case 7: // yellow
                    o.color = 3;
                    o.type = "round";
                    break;
                default: // TODO
                    o.color = -1;
                    o.type = "ignore";
                    break;
            }

            o.x = obj.position[0];
            if (std::isnan(o.x)) {
                o.x = 0.0;
            }

            o.y = obj.position[1];
            if (std::isnan(o.y)) {
                o.y = 0.0;
            }

            ma.obj_list.push_back(o);
        }
        catch (const std::out_of_range& oor) {
            usv_interfaces::msg::Object o;

            o.color = -1;
            o.x = 0;
            o.y = 0;
            o.type = "ignore";
            o.uuid = "";
            ma.obj_list.push_back(o);
        }
    }
    
    return ma;
}

rcl_interfaces::msg::SetParametersResult parametersCallback(
    const std::vector<rclcpp::Parameter>& parameters) {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;
    result.reason = "success";

    bool tracker_params_changed = false;
    double new_high_threshold = high_threshold;
    double new_low_threshold = low_threshold;
    int new_max_time_lost = max_time_lost;
    int new_min_hits = min_hits;

    for (const auto& param : parameters) {
        if (param.get_name() == "tracker_high_threshold") {
            new_high_threshold = param.as_double();
            tracker_params_changed = true;
        } else if (param.get_name() == "tracker_low_threshold") {
            new_low_threshold = param.as_double();
            tracker_params_changed = true;
        } else if (param.get_name() == "tracker_max_time_lost") {
            new_max_time_lost = param.as_int();
            tracker_params_changed = true;
        } else if (param.get_name() == "tracker_min_hits") {
            new_min_hits = param.as_int();
            tracker_params_changed = true;
        }
    }

    // Update tracker parameters if any changed
    if (tracker_params_changed) {
        RCLCPP_INFO(this->get_logger(), "Updating tracker parameters: high=%.2f, low=%.2f, max_lost=%d, min_hits=%d",
                    new_high_threshold, new_low_threshold, new_max_time_lost, new_min_hits);
        
        // Update member variables
        high_threshold = new_high_threshold;
        low_threshold = new_low_threshold;
        max_time_lost = new_max_time_lost;
        min_hits = new_min_hits;
        
        // Update the tracker with new parameters
        yolo_tracker.updateParameters(
            high_threshold,
            low_threshold,
            max_time_lost,
            min_hits
        );
    }

    return result;
}

public:
#ifdef USE_ZED_SDK
DetectorInterface()
    :   Node("bebblebrox_vision"), zed_interface(this->get_logger()) {
#else
DetectorInterface()
    :   Node("bebblebrox_vision") {
#endif

    /* PARAMETERS */
    
    // set simulation mode parameter
    this->declare_parameter("simulation_mode", false);
    simulation_mode = this->get_parameter("simulation_mode").as_bool();
    
    // camera intrinsics parameters for simulation
    if (simulation_mode) {
        this->declare_parameter("sim_fx", 500.0);
        this->declare_parameter("sim_fy", 500.0);
        this->declare_parameter("sim_cx", 320.0);
        this->declare_parameter("sim_cy", 240.0);
        this->declare_parameter("sim_width", 640);
        this->declare_parameter("sim_height", 480);
        
        fx = this->get_parameter("sim_fx").as_double();
        fy = this->get_parameter("sim_fy").as_double();
        cx = this->get_parameter("sim_cx").as_double();
        cy = this->get_parameter("sim_cy").as_double();
        img_width = this->get_parameter("sim_width").as_int();
        img_height = this->get_parameter("sim_height").as_int();
    }
    
    // final detections' topic
    this->declare_parameter("objects_yolo_topic", "/bebblebrox/objects/yolo");
    std::string objects_yolo_topic = this->get_parameter("objects_yolo_topic").as_string();

    this->declare_parameter("objects_shapes_topic", "/bebblebrox/objects/shapes");
    std::string objects_shapes_topic = this->get_parameter("objects_shapes_topic").as_string();
    
    // publisher's video topic
    this->declare_parameter("video_topic", "/bebblebrox/video");
    std::string video_topic = this->get_parameter("video_topic").as_string();
    
    // depth visualization topic
    this->declare_parameter("depth_viz_topic", "/bebblebrox/depth_colorized");
    std::string depth_viz_topic = this->get_parameter("depth_viz_topic").as_string();
    
    // subcriptions to detections' topic
    this->declare_parameter("yolo_sub_topic", "/yolo/detections");
    std::string yolo_sub_topic = this->get_parameter("yolo_sub_topic").as_string();

    this->declare_parameter("shapes_sub_topic", "/shapes/detections");
    std::string shapes_sub_topic = this->get_parameter("shapes_sub_topic").as_string();
    
    // pointcloud topic for simulation
    this->declare_parameter("pointcloud_topic", "/simulation/pointcloud");
    std::string pointcloud_topic = this->get_parameter("pointcloud_topic").as_string();

    // frame interval (ms)
    this->declare_parameter("frame_interval", 100);
    int frame_interval = this->get_parameter("frame_interval").as_int();

    // high threshold for bytetrack tracker
    this->declare_parameter("high_threshold", 0.6);
    this->high_threshold = this->get_parameter("high_threshold").as_double();

    // low threshold for bytetrack tracker
    this->declare_parameter("low_threshold", 0.1);
    this->low_threshold = this->get_parameter("low_threshold").as_double();

    // time lost in frames elapsed for bytetrack tracker
    this->declare_parameter("max_time_lost", 30);
    this->max_time_lost = this->get_parameter("max_time_lost").as_int();

    // min hits for bytetrack tracler
    this->declare_parameter("min_hits", 3);
    this->min_hits = this->get_parameter("min_hits").as_int();

    param_callback_handle_ = this->add_on_set_parameters_callback(
        std::bind(&DetectorInterface::parametersCallback, this, std::placeholders::_1)
    );

    // create bytetrack trackers with the specified parameters
    yolo_tracker = bytetrack::ByteTrack(
        high_threshold, 
        low_threshold, 
        max_time_lost, 
        min_hits, 
        this->get_logger()
    );

    /* PUBLISHERS */
    
    // yolo's final detections publisher
    this->yolo_pub = this->create_publisher<usv_interfaces::msg::ObjectList>(objects_yolo_topic, 10);

    // shapes' final detections publisher
    this->shapes_pub = this->create_publisher<usv_interfaces::msg::ObjectList>(objects_shapes_topic, 10);

    // video publisher
    rclcpp::Node::SharedPtr nh(std::shared_ptr<DetectorInterface>(this, [](auto *) {}));
    image_transport::ImageTransport it(nh);
    this->ip = it.advertise(video_topic, 5);
    
    // Depth visualization publisher
    this->depth_image_publisher = it.advertise(depth_viz_topic, 5);
    this->cam_info_pub = this->create_publisher<sensor_msgs::msg::CameraInfo>("/bebblebrox/camera_info", 10);

    /* SUBSCRIBERS */
    
    rclcpp::CallbackGroup::SharedPtr r_group;
    r_group = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    rclcpp::SubscriptionOptions options;
    options.callback_group = r_group;

    this->yolo_sub = this->create_subscription<usv_interfaces::msg::ZbboxArray>(
            yolo_sub_topic,
            rclcpp::SystemDefaultsQoS(),
            std::bind(&DetectorInterface::receive_yolo, this, _1),
            options
        );
    
    this->shapes_sub = this->create_subscription<usv_interfaces::msg::ZbboxArray>(
            shapes_sub_topic,
            rclcpp::SystemDefaultsQoS(),
            std::bind(&DetectorInterface::receive_shapes, this, _1),
            options
        );

    // subscribe to pointcloud for simulation mode
    if (simulation_mode) {
        this->pointcloud_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            pointcloud_topic,
            rclcpp::SystemDefaultsQoS(),
            std::bind(&DetectorInterface::receive_pointcloud, this, _1),
            options
        );
    }

    // in simulation mode, we don't need the timer since processing happens in the pointcloud callback
    if (!simulation_mode) {
        timer = this->create_wall_timer(
                std::chrono::milliseconds(frame_interval),
                std::bind(&DetectorInterface::frame_send, this)
            );
    }
    
    RCLCPP_INFO(this->get_logger(), "running in %s mode with organized pointcloud support", 
               simulation_mode ? "SIMULATION" : "REAL-LIFE");
    }
};

int main(int argc, char** argv) {
    cudaSetDevice(0);
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DetectorInterface>();
    rclcpp::executors::MultiThreadedExecutor exec;
    exec.add_node(node);
    exec.spin();
    rclcpp::shutdown();
    return 0;
}