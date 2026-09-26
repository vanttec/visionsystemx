from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
import os

def generate_launch_description():
    camera_model_arg = DeclareLaunchArgument(
        'camera_model',
        default_value='zed2i',
        description='ZED camera model to use: "zed" or "zed1" for ZED 1, "zed2i" for ZED 2i'
    )
    camera_model = LaunchConfiguration('camera_model')
    # Video source fallback: ZED SDK path only if explicitly requested,
    # otherwise USB webcam. Both publish /bebblebrox/video, so downstream
    # (yolo, lidar_range, dashboard republish) needs no changes.
    use_zed_arg = DeclareLaunchArgument(
        'use_zed',
        default_value='false',
        description='Use legacy ZED SDK video_feed instead of USB webcam'
    )
    use_zed = LaunchConfiguration('use_zed')
    velodyne = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('velodyne'),
                'launch',
                'velodyne-all-nodes-VLP16-composed-launch.py'
            ])
        ])
    )
    
    rqt = Node(
        package="rqt_gui",
        executable="rqt_gui"
    )
    
    rviz = Node(
        package="rviz2",
        executable="rviz2"
    )
    
    yolo_tensorrt = Node(
        package='visionsystemx',
        executable='yolo',
        name='yolo',
        output='screen',
        parameters=[
            {'engine_path': PathJoinSubstitution([
                FindPackageShare('visionsystemx'),
                'data',
                'SARASOTA.engine',
            ])},
            {'video_topic': '/bebblebrox/video'},
            {'output_topic': '/yolo/detections'},
            {'threshold': 0.2},
        ],
        # ros debug printing
        arguments=[
            '--ros-args', 
            '--log-level', 'DEBUG',
            '-p', 'image_transport.compressed.jpeg_quality:=60',
            '-p', 'image_transport.ffmpeg.preset:=ultrafast',
            '-p', 'image_transport.ffmpeg.tune:=zerolatency'
        ],
    )
    
    video_feed = Node(
        package='visionsystemx',
        executable='beeblebrox',
        name='beeblebrox',
        output='screen',
        condition=IfCondition(use_zed),
        parameters=[
            # Camera model: "zed" or "zed1" for ZED 1, "zed2i" for ZED 2i
            {'camera_model': camera_model},
            # Real-life mode flag
            {'simulation_mode': False},
            # Standard parameters
            {'video_topic': '/bebblebrox/video'},
            {'yolo_sub_topic': '/yolo/detections'},
            {'frame_interval': 100}, # run every 100 ms
        ],
        # arguments=['--ros-args', '--log-level', 'DEBUG']
    )

    # --- ZED replacement: USB webcam + LiDAR frustum ranging ---
    # Publishes /bebblebrox/video + /bebblebrox/video/camera_info.
    # Calibrate first (camera_calibration) and set camera_info_url.
    usb_cam = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='usb_cam',
        output='screen',
        condition=UnlessCondition(use_zed),
        parameters=[
            {'video_device': '/dev/video0'},
            {'framerate': 30.0},
            {'pixel_format': 'mjpeg2rgb'},
            {'camera_info_url': 'file:///tmp/webcam_calibration.yaml'},
            {'image_width': 1280},
            {'image_height': 720},
        ],
        remappings=[
            ('image_raw', '/bebblebrox/video'),
            ('camera_info', '/bebblebrox/video/camera_info'),
        ],
    )

    lidar_range = Node(
        package='visionsystemx',
        executable='lidar_bbox_range_node.py',
        name='lidar_bbox_range',
        output='screen',
        parameters=[
            {'lidar_topic': '/velodyne_points'},
            {'camera_info_topic': '/bebblebrox/video/camera_info'},
            {'image_topic': '/bebblebrox/video'},
            {'yolo_sub_topic': '/yolo/detections'},
            {'shapes_sub_topic': '/shapes/detections'},
            {'objects_yolo_topic': '/bebblebrox/objects/yolo'},
            {'objects_shapes_topic': '/bebblebrox/objects/shapes'},
        ],
    )

    fusion = Node(
        package='visionsystemx',
        executable='lidar_camera.py'
    )

    # Republish raw image as compressed for the web dashboard (rosbridge).
    # Subscribes /bebblebrox/video, which is published by EITHER video source
    # (beeblebrox with use_zed:=true, or usb_cam by default), so the dashboard
    # works under both with no extra fallback logic.
    image_republish = Node(
        package='image_transport',
        executable='republish',
        name='image_republish',
        arguments=['raw', 'compressed'],
        remappings=[
            ('in',              '/bebblebrox/video'),
            ('out/compressed',  '/bebblebrox/video/compressed'),
        ],
        output='screen',
    )

    return LaunchDescription([
        camera_model_arg,
        use_zed_arg,
        video_feed,
        usb_cam,
        velodyne,
        lidar_range,
        yolo_tensorrt,
        image_republish,
        # fusion,
        # rviz,
        # rqt,
    ])
