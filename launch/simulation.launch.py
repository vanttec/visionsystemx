from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node

def generate_launch_description():    
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
            {'engine_path': '/home/max/vanttec_usv/SARASOTA.engine'},
            {'video_topic': '/bebblebrox/video'},
            {'output_topic': '/bebblebrox/detections'},
            {'threshold': 0.5},
        ],
    )
    
    video_feed = Node(
        package='visionsystemx',
        executable='beeblebrox',
        name='beeblebrox',
        output='screen',
        parameters=[
            # simulation mode settings
            {'simulation_mode': True},
            
            # standard parameters
            {'video_topic': '/bebblebrox/video'},
            {'yolo_sub_topic': '/bebblebrox/detections'},
            
            # simulation-specific parameters
            {'pointcloud_topic': '/zed_rgbd/points'},
            {'depth_viz_topic': '/bebblebrox/points'},
            
            # optional camera parameters for simulation
            # these are used as fallback if the pointcloud doesn't have organized structure
            {'sim_width': 640},
            {'sim_height': 480},
            {'sim_fx': 500.0},
            {'sim_fy': 500.0},
            {'sim_cx': 320.0},
            {'sim_cy': 240.0},
        ],
        # arguments=['--ros-args', '--log-level', 'DEBUG']
    )
    
    return LaunchDescription([
        video_feed,
        yolo_tensorrt,
        # rviz,
    ])