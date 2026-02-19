from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
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
            {'engine_path': os.path.expanduser('~/vanttec_usv/src/visionsystemx/data/SARASOTA.engine')},
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

    fusion = Node(
        package='visionsystemx',
        executable='lidar_camera.py'
    )
    
    return LaunchDescription([
        # camera_model_arg,
        # video_feed,
        yolo_tensorrt,
        # velodyne,
        # fusion,
        # rviz,
        # rqt,
    ])
