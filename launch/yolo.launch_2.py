
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node

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
    
    shared_video_topic = '/bebblebrox/video'
    
    # --- Instance 1: Primary (All buoys) ---
    yolo_primary = Node(
        package='visionsystemx',
        executable='yolo',
        name='yolo_node',
        namespace='yolo',
        output='screen',
        parameters=[
            {'engine_path': '/home/asv/vanttec_usv/src/visionsystemx/data/SARASOTA.engine'},
            {'threshold': 0.5}, # High threshold
        ],
        remappings=[
            ('video', shared_video_topic), # Redirects local 'video' to the shared global topic
        ]
    )

    # --- Instance 2: Secondary (Light indicator) ---
    yolo_secondary = Node(
        package='visionsystemx',
        executable='yolo',
        name='yolo_node',
        namespace='yolo_secondary',
        output='screen',
        parameters=[
            {'engine_path': '/home/asv/vanttec_usv/src/visionsystemx/data/indicatoryolo8.engine'},
            {'threshold': 0.5}, # Low threshold to catch faint objects
        ],
        remappings=[
            ('video', shared_video_topic), # Also points to the same shared topic
        ]
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
        yolo_primary,
        yolo_secondary,
        # velodyne,
        # fusion,
        # rviz,
        # rqt,
    ])
