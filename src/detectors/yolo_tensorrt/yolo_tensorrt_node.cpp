#include "rclcpp/rclcpp.hpp"
#include "image_transport/image_transport.hpp"

#include "sensor_msgs/msg/image.hpp"
#include "usv_interfaces/msg/zbbox_array.hpp"
#include "usv_interfaces/msg/zbbox.hpp"

#include "cv_bridge/cv_bridge.h"
#include "opencv2/opencv.hpp"

#include "yolo_tensorrt.hpp"

using std::placeholders::_1;

const std::vector<std::string> NAMES {
	"black_buoy",
	"blue_buoy",
	"course_marker",
    "green_buoy",
    "port_marker",
    "red_buoy",
    "starboard_marker",
    "yellow_marker",
    "blue_square",
	"blue_triangle",
	"green_circle",
	"green_cross",
	"green_square",
	"green_triangle",
	"red_circle",
	"red_cross",
	"red_square",
	"red_triangle",
    "blue_circle",
    "blue_cross"
};

const std::vector<std::vector<unsigned int>> COLORS {
	{0, 0, 0},
	{0, 0, 0},
	{0, 0, 0},
	{0, 0, 0},
	{0, 0, 0},
	{0, 0, 0},
	{0, 0, 0},
	{0, 0, 0},
	{0, 0, 0},
	{0, 0, 0},
	{0, 0, 0},
	{0, 0, 0},
	{0, 0, 0},
	{0, 0, 0},
	{0, 0, 0},
	{0, 0, 0},
	{0, 0, 0},
	{0, 0, 0},
	{0, 0, 0},
	{0, 0, 0},
	{0, 0, 0}
};

template <typename E>
class YoloDetector : public rclcpp::Node {
private:
    cv::Size size = cv::Size{640, 640};

    std::string engine_path;
    std::string video_topic;
    std::string output_topic;
    double threshold;

    std::unique_ptr<E> detector_engine;

    // image transport objects
    std::shared_ptr<image_transport::ImageTransport> it;
    std::shared_ptr<image_transport::Subscriber> is;
    image_transport::Publisher draw_pub;  // publisher for the drawn image

    // publisher for detections
    std::shared_ptr<rclcpp::Publisher<usv_interfaces::msg::ZbboxArray>> dets;

    void frame(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
    {
        try {

            RCLCPP_DEBUG(this->get_logger(), "Image encoding: %s", msg->encoding.c_str());

            // convert incoming ROS image to OpenCV image
            auto cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
            cv::Mat img = cv_ptr->image;

            if (img.empty() || img.cols <= 0 || img.rows <= 0) {
                RCLCPP_ERROR(this->get_logger(), "Empty or invalid image dimensions: %dx%d", 
                            img.cols, img.rows);
                return;
            }

            // run inference using YOLOv8
            detector_engine->copy_from_Mat(img, size);
            detector_engine->infer();

            // get detection results
            usv_interfaces::msg::ZbboxArray objs = detector_engine->postprocess();
            objs.header.stamp = this->now();

            // publish detections
            this->dets->publish(objs);
            RCLCPP_INFO(this->get_logger(), "--> inference done [%ld detections]", objs.boxes.size());

            // draw the detections on the image
            cv::Mat annotated;
            detector_engine->draw_objects(img, annotated, objs, NAMES, COLORS);

            // convert the annotated image back to a ROS image message
            auto annotated_msg = cv_bridge::CvImage(msg->header, "bgr8", annotated).toImageMsg();
            this->draw_pub.publish(annotated_msg);
        } catch (const std::exception &e) {
            RCLCPP_ERROR(this->get_logger(), "Error during inference: %s", e.what());
        }
    }

public:
    YoloDetector() : Node("yolo")
    {
        this->declare_parameter("engine_path", "/home/asv/vanttec_usv/src/visionsystemx/data/SARASOTA.engine");
        engine_path = this->get_parameter("engine_path").as_string();

        this->declare_parameter("video_topic", "/bebblebrox/video");
        video_topic = this->get_parameter("video_topic").as_string();

        this->declare_parameter("output_topic", "/yolo/detections");
        output_topic = this->get_parameter("output_topic").as_string();

        this->declare_parameter("threshold", 0.1);
        threshold = this->get_parameter("threshold").as_double();

        size = cv::Size{640, 640};

        try {
            detector_engine = std::make_unique<E>(engine_path, threshold, this->get_logger());
            detector_engine->make_pipe(true);
        } catch (const std::exception &e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to initialize YOLO detector: %s", e.what());
            throw;
        }

        // publisher for detection results
        this->dets = this->create_publisher<usv_interfaces::msg::ZbboxArray>(output_topic, 10);
    }

    void init() {
        // initialize image transport
        this->it = std::make_shared<image_transport::ImageTransport>(this->shared_from_this());

        // subscriber to the raw video topic
        this->is = std::make_shared<image_transport::Subscriber>(
            it->subscribe(
                video_topic,
                10,
                std::bind(&YoloDetector::frame, this, _1)
            )
        );

        // advertise the topic for the drawn (annotated) image
        this->draw_pub = it->advertise("yolo/draw", 10);

        RCLCPP_INFO(this->get_logger(), "-> yolo ready");
    }
};

int main(int argc, char **argv)
{
    try {
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            std::cerr << "CUDA device selection failed: " << cudaGetErrorString(err) << std::endl;
            return 1;
        }

        rclcpp::init(argc, argv);
        auto node = std::make_shared<YoloDetector<YOLOv8>>();
        node->init();
        rclcpp::spin(node);
        rclcpp::shutdown();
    } catch (const std::exception &e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
