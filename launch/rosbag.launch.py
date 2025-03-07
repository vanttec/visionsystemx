from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node

def generate_launch_description():
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
            {'engine_path': '/home/vanttec/vanttec_usv/SARASOTA.engine'},
            {'video_topic': '/bebblebrox/video'},
            {'output_topic': '/shapes/detections'},
            {'threshold': 0.60},
        ],
        # ros debug printing
        # arguments=['--ros-args', '--log-level', 'DEBUG']
    )
    
    video_feed = Node(
        package='visionsystemx',
        executable='beeblebrox',
        name='beeblebrox',
        output='screen',
        parameters=[
            # Real-life mode flag
            {'simulation_mode': False},
            # Standard parameters
            {'objects_shapes_topic': '/objects_docking'},
            {'video_topic': '/none'},
            {'shapes_sub_topic': '/shapes/detections'},
            {'frame_interval': 100}, # run every 100 ms  
        ],
        # arguments=['--ros-args', '--log-level', 'DEBUG']
    )

    fusion = Node(
        package='visionsystemx',
        executable='lidar_camera.py'
    )
    
    return LaunchDescription([
        video_feed,
        yolo_tensorrt,
        # velodyne,
        # fusion,
        # rviz,
        rqt,
    ])
