#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

class LidarCameraProjectionNode : public rclcpp::Node
{
public:
    LidarCameraProjectionNode() : Node("lidar_camera_projection_node")
    {
        // Define extrinsic matrix (LiDAR to camera transformation)
        T_lidar_to_cam_ = Eigen::Matrix4d::Identity();
        // Specific transformation matrix
        T_lidar_to_cam_ << 0.004925697387197947, -0.9999744448207916,   -0.0051814293973278525,  0.06684500162782157,
                          0.01300526353504694,   0.00524511394559779,  -0.9999016711157561,    -0.1345418740635012,
                          0.99990329563695,      0.004857827194069242,  0.013030767027264023,  -0.12626667685463364,
                          0.0,                   0.0,                   0.0,                    1.0;

        // Define camera intrinsic parameters
        camera_matrix_ = cv::Mat::zeros(3, 3, CV_64F);
        camera_matrix_.at<double>(0, 0) = 527.865; // fx
        camera_matrix_.at<double>(0, 2) = 658.685; // cx
        camera_matrix_.at<double>(1, 1) = 527.38;  // fy
        camera_matrix_.at<double>(1, 2) = 363.345; // cy
        camera_matrix_.at<double>(2, 2) = 1.0;

        // Define distortion coefficients (zeros for no distortion)
        dist_coeffs_ = cv::Mat::zeros(1, 5, CV_64F);

        // Log the parameters
        RCLCPP_INFO(this->get_logger(), "Loaded extrinsic matrix:");
        for (int i = 0; i < 4; i++) {
            RCLCPP_INFO(this->get_logger(), "[%f, %f, %f, %f]",
                T_lidar_to_cam_(i, 0), T_lidar_to_cam_(i, 1),
                T_lidar_to_cam_(i, 2), T_lidar_to_cam_(i, 3));
        }
        
        RCLCPP_INFO(this->get_logger(), "Camera matrix:");
        for (int i = 0; i < 3; i++) {
            RCLCPP_INFO(this->get_logger(), "[%f, %f, %f]",
                camera_matrix_.at<double>(i, 0),
                camera_matrix_.at<double>(i, 1),
                camera_matrix_.at<double>(i, 2));
        }
        
        RCLCPP_INFO(this->get_logger(), "Distortion coeffs:");
        for (int i = 0; i < 5; i++) {
            RCLCPP_INFO(this->get_logger(), "%f", dist_coeffs_.at<double>(0, i));
        }

        // Define topics
        std::string lidar_topic = "/velodyne_points";
        std::string image_topic = "/bebblebrox/video";
        std::string projected_topic = "/bebblebrox/video/projected";

        RCLCPP_INFO(this->get_logger(), "Subscribing to lidar topic: %s", lidar_topic.c_str());
        RCLCPP_INFO(this->get_logger(), "Subscribing to image topic: %s", image_topic.c_str());

        // Create subscribers with message filters
        image_sub_.subscribe(this, image_topic);
        lidar_sub_.subscribe(this, lidar_topic);

        // Create synchronizer with approximate time policy
        sync_ = std::make_shared<Sync>(SyncPolicy(5), image_sub_, lidar_sub_);
        
        // Set max interval duration (slop)
        static_cast<SyncPolicy*>(sync_->getPolicy())->setMaxIntervalDuration(rclcpp::Duration::from_seconds(0.07));
        sync_->registerCallback(
            std::bind(&LidarCameraProjectionNode::sync_callback, this, 
                     std::placeholders::_1, std::placeholders::_2));

        // Create publisher
        pub_image_ = this->create_publisher<sensor_msgs::msg::Image>(projected_topic, 1);

        skip_rate_ = 1; // Skip rate for point cloud processing
    }

private:
    void sync_callback(const sensor_msgs::msg::Image::ConstSharedPtr& image_msg,
                      const sensor_msgs::msg::PointCloud2::ConstSharedPtr& lidar_msg)
    {
        try {
            // Convert ROS image to OpenCV image
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(image_msg, "bgr8");
            cv::Mat cv_image = cv_ptr->image;

            // Extract points from PointCloud2
            std::vector<Eigen::Vector3d> xyz_lidar;
            pointcloud2_to_xyz_array(lidar_msg, xyz_lidar, skip_rate_);

            size_t n_points = xyz_lidar.size();
            if (n_points == 0) {
                RCLCPP_WARN(this->get_logger(), "Empty cloud. Nothing to project.");
                publish_image(cv_image, image_msg->header);
                return;
            }

            // Transform and project points to the image
            std::vector<cv::Point2f> image_points;
            transform_and_project_points(xyz_lidar, image_points);

            if (image_points.empty()) {
                RCLCPP_INFO(this->get_logger(), "No points in front of camera (z>0).");
                publish_image(cv_image, image_msg->header);
                return;
            }

            // Draw points on image
            int h = cv_image.rows;
            int w = cv_image.cols;
            for (const auto& point : image_points) {
                int u_int = static_cast<int>(point.x + 0.5);
                int v_int = static_cast<int>(point.y + 0.5);
                if (0 <= u_int && u_int < w && 0 <= v_int && v_int < h) {
                    cv::circle(cv_image, cv::Point(u_int, v_int), 2, cv::Scalar(0, 255, 0), -1);
                }
            }

            // Publish the image
            publish_image(cv_image, image_msg->header);
        }
        catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "CV bridge exception: %s", e.what());
        }
    }

    void pointcloud2_to_xyz_array(
        const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud_msg,
        std::vector<Eigen::Vector3d>& points,
        int skip_rate)
    {
        points.clear();

        if (cloud_msg->height == 0 || cloud_msg->width == 0) {
            RCLCPP_WARN(this->get_logger(), "Empty point cloud received");
            return;
        }

        // Check if point cloud has x, y, z fields
        int x_idx = -1, y_idx = -1, z_idx = -1;
        for (size_t i = 0; i < cloud_msg->fields.size(); ++i) {
            if (cloud_msg->fields[i].name == "x") x_idx = i;
            if (cloud_msg->fields[i].name == "y") y_idx = i;
            if (cloud_msg->fields[i].name == "z") z_idx = i;
        }

        if (x_idx == -1 || y_idx == -1 || z_idx == -1) {
            RCLCPP_ERROR(this->get_logger(), "Point cloud is missing x, y, or z fields");
            return;
        }

        // Get field offsets
        size_t x_offset = cloud_msg->fields[x_idx].offset;
        size_t y_offset = cloud_msg->fields[y_idx].offset;
        size_t z_offset = cloud_msg->fields[z_idx].offset;

        // Extract points from the cloud
        const uint8_t* cloud_data = cloud_msg->data.data();
        size_t point_step = cloud_msg->point_step;
        size_t row_step = cloud_msg->row_step;
        size_t num_points = cloud_msg->width * cloud_msg->height;

        points.reserve(num_points / skip_rate);

        for (size_t i = 0; i < num_points; i += skip_rate) {
            size_t offset = i * point_step;
            float x = *reinterpret_cast<const float*>(&cloud_data[offset + x_offset]);
            float y = *reinterpret_cast<const float*>(&cloud_data[offset + y_offset]);
            float z = *reinterpret_cast<const float*>(&cloud_data[offset + z_offset]);

            // Skip NaN or inf values
            if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z)) {
                points.emplace_back(x, y, z);
            }
        }
    }

    void transform_and_project_points(
        const std::vector<Eigen::Vector3d>& xyz_lidar,
        std::vector<cv::Point2f>& image_points)
    {
        image_points.clear();
        
        // Transform points from LiDAR to camera frame
        std::vector<cv::Point3d> xyz_cam_cv;
        xyz_cam_cv.reserve(xyz_lidar.size());
        
        for (const auto& pt_lidar : xyz_lidar) {
            // Convert to homogeneous coordinates
            Eigen::Vector4d pt_lidar_h(pt_lidar.x(), pt_lidar.y(), pt_lidar.z(), 1.0);
            
            // Apply transformation
            Eigen::Vector4d pt_cam_h = T_lidar_to_cam_ * pt_lidar_h;
            
            // Convert back to 3D coordinates
            double x = pt_cam_h(0);
            double y = pt_cam_h(1);
            double z = pt_cam_h(2);
            
            // Only keep points in front of the camera (z > 0)
            if (z > 0.0) {
                xyz_cam_cv.emplace_back(x, y, z);
            }
        }
        
        if (xyz_cam_cv.empty()) {
            return;
        }
        
        // Project 3D points to 2D image points
        cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);  // No rotation
        cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);  // No translation
        
        cv::projectPoints(xyz_cam_cv, rvec, tvec, camera_matrix_, dist_coeffs_, image_points);
    }

    void publish_image(const cv::Mat& image, const std_msgs::msg::Header& header)
    {
        cv_bridge::CvImage cv_bridge_image;
        cv_bridge_image.header = header;
        cv_bridge_image.encoding = "bgr8";
        cv_bridge_image.image = image;
        
        pub_image_->publish(*cv_bridge_image.toImageMsg());
    }

    // Typedefs for message filters
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<
                        sensor_msgs::msg::Image, 
                        sensor_msgs::msg::PointCloud2>;
    using Sync = message_filters::Synchronizer<SyncPolicy>;

    // Subscribers and publisher
    message_filters::Subscriber<sensor_msgs::msg::Image> image_sub_;
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> lidar_sub_;
    std::shared_ptr<Sync> sync_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_image_;

    // Parameters
    Eigen::Matrix4d T_lidar_to_cam_;
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    int skip_rate_;
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<LidarCameraProjectionNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}