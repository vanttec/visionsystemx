#include "rclcpp/rclcpp.hpp"
#include "image_transport/image_transport.hpp"

#include "usv_interfaces/msg/zbbox_array.hpp"
#include "usv_interfaces/msg/zbbox.hpp"

#include "cv_bridge/cv_bridge.h"
#include "opencv2/opencv.hpp"

#include "yolo_tensorrt.hpp"

using std::placeholders::_1;

const std::vector<std::string> NAMES {
	"black_buoy",
	"blue_bbuoy",
	"course_marker",
	"green_buoy",
	"port_marker",
	"red_buoy",
	"starboard_marker",
	"yellow_marker",
	"ducks",
	"blue_cross",
	"red_circle",
	"red_triangle",
	"green_cross",
	"blue_triangle",
	"green_square",
	"red_cross",
	"green_circle",
	"red_square",
	"green_triangle",
	"blue_circle",
	"blue_square"
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
class YoloDetector: public rclcpp::Node {

private:
	cv::Size		size        = cv::Size{640, 640};

	std::string		engine_path;
	std::string		video_topic;
	std::string		output_topic;
	std::string		output_image_topic;
	double			threshold;

	E*			detector_engine;

	std::shared_ptr<image_transport::ImageTransport> it;
	std::shared_ptr<image_transport::Subscriber> is;
	std::shared_ptr<image_transport::Publisher> idp;
	std::shared_ptr<rclcpp::Publisher<usv_interfaces::msg::ZbboxArray>> dets;


	// rclcpp::Publisher<usv_interfaces::msg::ZbboxArray>::SharedPtr dets;



void frame(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
	auto img = cv_bridge::toCvCopy(msg, "bgr8")->image;

	detector_engine->copy_from_Mat(img, size);
	detector_engine->infer();
	
	usv_interfaces::msg::ZbboxArray objs;
	objs = detector_engine->postprocess();

	objs.header.stamp = this->now();

	this->dets->publish( objs );

	RCLCPP_INFO(this->get_logger(), "--> inference done [%ld]", objs.boxes.size());

	detector_engine->draw_objects(img, img, objs, NAMES, COLORS);

	std_msgs::msg::Header hdr;
	sensor_msgs::msg::Image::SharedPtr detmsg = cv_bridge::CvImage(hdr, "bgr8", img).toImageMsg();
	detmsg->header.stamp = this->get_clock()->now();

	this->idp->publish( detmsg );
}

public:
	YoloDetector() : Node("yolo") {

		this->declare_parameter("engine_path", "/home/vanttec/vanttec_usv/SARASOTA.engine");
		engine_path = this->get_parameter("engine_path").as_string();

		this->declare_parameter("video_topic", "/beeblebrox/video");
		video_topic = this->get_parameter("video_topic").as_string();

		this->declare_parameter("output_topic", "/yolo/detections");
		output_topic = this->get_parameter("output_topic").as_string();

		this->declare_parameter("output_image_topic", "/yolo/detections/image");
		output_image_topic = this->get_parameter("output_image_topic").as_string();

		this->declare_parameter("threshold", 0.6);
		threshold = this->get_parameter("threshold").as_double();

		size = cv::Size{640, 640};

		detector_engine = new YOLOv8(engine_path, threshold, this->get_logger());
		
		detector_engine->make_pipe(true);

		this->dets = this->create_publisher<usv_interfaces::msg::ZbboxArray>(output_topic, 10);
	}

	void init() {
		this->it = std::make_shared<image_transport::ImageTransport>(this->shared_from_this());

		this->is = std::make_shared<image_transport::Subscriber>(
			it->subscribe(
				video_topic,
				10,
				std::bind(&YoloDetector::frame, this, std::placeholders::_1)
			)
		);

		this->idp = std::make_shared<image_transport::Publisher>(
			it->advertise(
				output_image_topic,
				5
			)
		);

		RCLCPP_INFO(this->get_logger(), "-> yolo ready");
	}

};


int main(int argc, char **argv)
{
	cudaSetDevice(0);

	rclcpp::init(argc, argv);

	auto node = std::make_shared< YoloDetector<YOLOv8> >();
	node->init();

	rclcpp::spin(node);

	rclcpp::shutdown();
	
	return 0;
}
