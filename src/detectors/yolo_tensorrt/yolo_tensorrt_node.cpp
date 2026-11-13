#include "rclcpp/rclcpp.hpp"
#include "image_transport/image_transport.hpp"

#include "sensor_msgs/msg/image.hpp"
#include "usv_interfaces/msg/zbbox_array.hpp"
#include "usv_interfaces/msg/zbbox.hpp"

#include "cv_bridge/cv_bridge.h"
#include "opencv2/opencv.hpp"

#include "yolo_tensorrt.hpp"

#include <chrono>
#include <iterator>
#include <limits>

using std::placeholders::_1;

const std::vector<std::string> DEFAULT_CLASS_NAMES {
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

std::vector<std::vector<unsigned int>> make_colors(size_t count)
{
    std::vector<std::vector<unsigned int>> colors;
    colors.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        // Deterministic BGR palette that remains readable as models are added.
        colors.push_back({
            static_cast<unsigned int>((37 * i + 97) % 256),
            static_cast<unsigned int>((17 * i + 191) % 256),
            static_cast<unsigned int>((29 * i + 53) % 256),
        });
    }
    return colors;
}

template <typename E>
class YoloDetector : public rclcpp::Node {
private:
    struct ModelInstance {
        std::string name;
        std::string engine_path;
        int label_offset;
        std::unique_ptr<E> engine;
    };

    std::string engine_path;
    std::string video_topic;
    std::string output_topic;
    double threshold;
    bool strict_model_loading;

    std::vector<ModelInstance> models;
    std::vector<std::string> class_names;
    std::vector<std::vector<unsigned int>> colors;

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

            usv_interfaces::msg::ZbboxArray objs;
            objs.header = msg->header;
            objs.header.stamp = this->now();

            const auto frame_start = std::chrono::steady_clock::now();
            for (auto& model : models) {
                const auto model_start = std::chrono::steady_clock::now();
                try {
                    // Each engine determines its own static input dimensions.
                    model.engine->copy_from_Mat(img);
                    model.engine->infer();
                    auto model_objects = model.engine->postprocess();

                    for (auto& box : model_objects.boxes) {
                        box.label += model.label_offset;
                    }

                    const size_t model_detection_count = model_objects.boxes.size();
                    objs.boxes.insert(
                        objs.boxes.end(),
                        std::make_move_iterator(model_objects.boxes.begin()),
                        std::make_move_iterator(model_objects.boxes.end()));

                    const auto model_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - model_start).count();
                    RCLCPP_DEBUG(
                        this->get_logger(),
                        "Model '%s': %.2f ms, %zu detections",
                        model.name.c_str(),
                        model_ms,
                        model_detection_count);
                } catch (const std::exception& e) {
                    RCLCPP_ERROR(
                        this->get_logger(),
                        "Inference failed for model '%s': %s",
                        model.name.c_str(),
                        e.what());
                }
            }

            const auto frame_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - frame_start).count();

            // publish detections
            this->dets->publish(objs);
            RCLCPP_DEBUG(
                this->get_logger(),
                "All models: %.2f ms, %zu detections",
                frame_ms,
                objs.boxes.size());

            // draw the detections on the image
            cv::Mat annotated;
            models.front().engine->draw_objects(img, annotated, objs, class_names, colors);

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
        this->declare_parameter("engine_path", "/home/asv/vanttec_usv/src/visionsystemx/data/CORE.engine");
        engine_path = this->get_parameter("engine_path").as_string();

        this->declare_parameter("video_topic", "/bebblebrox/video");
        video_topic = this->get_parameter("video_topic").as_string();

        this->declare_parameter("output_topic", "/yolo/detections");
        output_topic = this->get_parameter("output_topic").as_string();

        this->declare_parameter("threshold", 0.1);
        threshold = this->get_parameter("threshold").as_double();

        this->declare_parameter<std::vector<std::string>>(
            "engine_paths", std::vector<std::string>{});
        auto engine_paths = this->get_parameter("engine_paths").as_string_array();
        if (engine_paths.empty()) {
            engine_paths.push_back(engine_path);
        }

        this->declare_parameter<std::vector<std::string>>(
            "model_names", std::vector<std::string>{});
        auto model_names = this->get_parameter("model_names").as_string_array();
        if (model_names.empty()) {
            for (size_t i = 0; i < engine_paths.size(); ++i) {
                model_names.push_back(i == 0 ? "base" : "model_" + std::to_string(i));
            }
        }
        if (model_names.size() != engine_paths.size()) {
            throw std::invalid_argument("model_names must have one entry per engine_paths entry");
        }

        this->declare_parameter<std::vector<int64_t>>(
            "label_offsets", std::vector<int64_t>{});
        auto label_offsets = this->get_parameter("label_offsets").as_integer_array();
        if (label_offsets.empty() && engine_paths.size() == 1) {
            label_offsets.push_back(0);
        }
        if (label_offsets.size() != engine_paths.size()) {
            throw std::invalid_argument(
                "label_offsets must have one entry per engine when multiple models are configured");
        }
        for (const auto offset : label_offsets) {
            if (offset < 0 || offset > std::numeric_limits<int>::max()) {
                throw std::invalid_argument("label_offsets entries must fit in a non-negative int");
            }
        }

        this->declare_parameter<std::vector<double>>(
            "model_thresholds", std::vector<double>{});
        auto model_thresholds = this->get_parameter("model_thresholds").as_double_array();
        if (model_thresholds.empty()) {
            model_thresholds.assign(engine_paths.size(), threshold);
        }
        if (model_thresholds.size() != engine_paths.size()) {
            throw std::invalid_argument("model_thresholds must have one entry per engine_paths entry");
        }
        for (const auto model_threshold : model_thresholds) {
            if (model_threshold < 0.0 || model_threshold > 1.0) {
                throw std::invalid_argument("model_thresholds entries must be between 0 and 1");
            }
        }

        this->declare_parameter<std::vector<std::string>>("class_names", DEFAULT_CLASS_NAMES);
        class_names = this->get_parameter("class_names").as_string_array();
        colors = make_colors(class_names.size());

        this->declare_parameter("strict_model_loading", true);
        strict_model_loading = this->get_parameter("strict_model_loading").as_bool();

        for (size_t i = 0; i < engine_paths.size(); ++i) {
            try {
                auto detector_engine = std::make_unique<E>(
                    engine_paths[i], model_thresholds[i], this->get_logger());
                detector_engine->make_pipe(true);
                models.push_back(ModelInstance{
                    model_names[i],
                    engine_paths[i],
                    static_cast<int>(label_offsets[i]),
                    std::move(detector_engine),
                });
                RCLCPP_INFO(
                    this->get_logger(),
                    "Loaded model '%s' from %s with label offset %ld",
                    model_names[i].c_str(),
                    engine_paths[i].c_str(),
                    label_offsets[i]);
            } catch (const std::exception& e) {
                RCLCPP_ERROR(
                    this->get_logger(),
                    "Failed to initialize model '%s' from %s: %s",
                    model_names[i].c_str(),
                    engine_paths[i].c_str(),
                    e.what());
                if (strict_model_loading || i == 0) {
                    throw;
                }
            }
        }

        if (models.empty()) {
            throw std::runtime_error("No YOLO models were loaded");
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

        RCLCPP_INFO(this->get_logger(), "-> yolo ready with %zu model(s)", models.size());
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
        auto node = std::make_shared<YoloDetector<YOLO26>>();
        node->init();
        rclcpp::spin(node);
        rclcpp::shutdown();
    } catch (const std::exception &e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
