#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image, PointField
import numpy as np
import struct
from cv_bridge import CvBridge
import cv2

class PointCloudProjector(Node):
    def __init__(self):
        super().__init__('pointcloud_projector')
        
        # Create subscriber for organized PointCloud2
        self.pointcloud_subscription = self.create_subscription(
            PointCloud2,
            '/zed_rgbd/points',  # Replace with your actual pointcloud topic
            self.pointcloud_callback,
            10)
        
        # Create publisher for the projected image
        self.depth_image_publisher = self.create_publisher(
            Image,
            '/proj_depth',
            10)
        
        # Create publisher for RGB visualization (optional)
        self.rgb_vis_publisher = self.create_publisher(
            Image,
            '/proj_rgb',
            10)
        
        self.cv_bridge = CvBridge()
        self.get_logger().info('PointCloud Projector has been initialized')
    
    def pointcloud_to_array(self, cloud_msg):
        """
        Convert a PointCloud2 message to a structured numpy array
        """
        # Get cloud dimensions
        width = cloud_msg.width
        height = cloud_msg.height
        point_step = cloud_msg.point_step
        row_step = cloud_msg.row_step
        
        # Check if the pointcloud is organized
        if height <= 1:
            self.get_logger().error("This pointcloud is not organized! Cannot project to image.")
            return None
        
        # Get field offsets for x, y, z (and rgb if available)
        x_offset = y_offset = z_offset = rgb_offset = None
        has_rgb = False
        
        for field in cloud_msg.fields:
            if field.name == 'x':
                x_offset = field.offset
            elif field.name == 'y':
                y_offset = field.offset
            elif field.name == 'z':
                z_offset = field.offset
            elif field.name in ('rgb', 'rgba'):
                rgb_offset = field.offset
                has_rgb = True
        
        # Create numpy arrays to hold the data
        depth_image = np.zeros((height, width), dtype=np.float32)
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8) if has_rgb else None
        
        # Maximum depth value for visualization (adjust as needed)
        max_depth = 10.0  # 10 meters
        
        # Extract data from the pointcloud
        for i in range(height):
            for j in range(width):
                # Calculate the index in the raw data
                point_index = i * row_step + j * point_step
                
                # Extract x, y, z values
                x = struct.unpack_from('f', cloud_msg.data, point_index + x_offset)[0]
                y = struct.unpack_from('f', cloud_msg.data, point_index + y_offset)[0]
                z = struct.unpack_from('f', cloud_msg.data, point_index + z_offset)[0]
                
                # Store depth (z value) in the depth image
                # Handle invalid points (NaN or infinity)
                if not np.isfinite(z):
                    depth_image[i, j] = 0.0  # Mark as invalid
                else:
                    depth_image[i, j] = z
                
                # Extract RGB if available
                if has_rgb and rgb_image is not None:
                    rgb = struct.unpack_from('I', cloud_msg.data, point_index + rgb_offset)[0]
                    # Convert RGB from pointcloud format to OpenCV format
                    b = (rgb & 0x000000FF)
                    g = (rgb & 0x0000FF00) >> 8
                    r = (rgb & 0x00FF0000) >> 16
                    rgb_image[i, j] = [r, g, b]
        
        return depth_image, rgb_image
    
    def pointcloud_callback(self, msg):
        # Check if the cloud is organized
        if msg.height <= 1:
            self.get_logger().error("Received an unorganized pointcloud. Cannot project to image.")
            return
            
        self.get_logger().info(f"Processing organized pointcloud with shape {msg.width}x{msg.height}")
        
        # Convert pointcloud to depth and RGB images
        depth_image, rgb_image = self.pointcloud_to_array(msg)
        
        if depth_image is None:
            self.get_logger().error("Failed to convert pointcloud to image")
            return
        
        # Normalize depth image for better visualization (0 to 255)
        max_depth = 10.0  # Adjust based on your data
        depth_vis = np.clip(depth_image, 0, max_depth)
        depth_vis_normalized = (depth_vis / max_depth * 255).astype(np.uint8)
        
        # Apply colormap to depth image for better visualization
        depth_colormap = cv2.applyColorMap(depth_vis_normalized, cv2.COLORMAP_JET)
        
        # Create ROS Image messages
        depth_msg = self.cv_bridge.cv2_to_imgmsg(depth_image, encoding="32FC1")
        depth_msg.header = msg.header
        
        depth_vis_msg = self.cv_bridge.cv2_to_imgmsg(depth_colormap, encoding="bgr8")
        depth_vis_msg.header = msg.header
        
        # Publish depth image
        self.depth_image_publisher.publish(depth_msg)
        
        # Publish RGB visualization if available
        if rgb_image is not None:
            rgb_msg = self.cv_bridge.cv2_to_imgmsg(rgb_image, encoding="rgb8")
            rgb_msg.header = msg.header
            self.rgb_vis_publisher.publish(rgb_msg)
        else:
            # If no RGB data, publish the colorized depth instead
            self.rgb_vis_publisher.publish(depth_vis_msg)
            
        self.get_logger().info("Published projected images")

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudProjector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node stopped cleanly')
    except Exception as e:
        node.get_logger().error(f'Exception in node: {str(e)}')
    finally:
        # Destroy the node explicitly
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()