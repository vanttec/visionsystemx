#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import torch
from PIL import Image as PILImage
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


class GreenSquareDetectorNode(Node):
    def __init__(self):
        super().__init__('green_square_detector')
        
        # Declare parameters
        self.declare_parameter('box_threshold', 0.4)
        self.declare_parameter('text_threshold', 0.3)
        
        # Get parameters
        self.box_threshold = self.get_parameter('box_threshold').value
        self.text_threshold = self.get_parameter('text_threshold').value
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Load model
        self.get_logger().info('Loading object detection model...')
        model_id = "IDEA-Research/grounding-dino-tiny"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        self.device = device
        self.get_logger().info(f'Model loaded, using device: {device}')
        
        # Create publisher for detection results
        self.detection_publisher = self.create_publisher(
            Bool, 
            '/usv/mission/green_light', 
            10
        )
        
        # Create subscription to camera images
        self.image_subscription = self.create_subscription(
            Image,
            '/bebblebrox/video',
            self.image_callback,
            10
        )
        
        self.get_logger().info('Green square detector node initialized')
    
    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            
            # Convert to PIL Image
            pil_image = PILImage.fromarray(cv_image)
            
            # Text prompt for zero-shot detection - MUST be lowercase and end with a dot
            text = "a green square."
            
            # Process image with model
            inputs = self.processor(images=pil_image, text=text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Post-process results
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[pil_image.size[::-1]]
            )
            
            # Check if a green square was detected
            detected = False
            if len(results[0]['boxes']) > 0:
                detected = True
                self.get_logger().info('Green square detected!')
            
            # Publish detection result
            detection_msg = Bool()
            detection_msg.data = detected
            self.detection_publisher.publish(detection_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    node = GreenSquareDetectorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()