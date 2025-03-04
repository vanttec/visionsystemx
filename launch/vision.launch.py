from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    return LaunchDescription([
        Node(
            package= 'visionsystemx',
            executable='beeblebrox',
            name='beeblebrox',
            output='screen',
            parameters=[
                {'objects_shapes_topic': '/objects_docking'},
                {'video_topic': '/bebblebrox/video'},
                {'shapes_sub_topic': '/shapes/detections'},
                {'frame_interval': 100}, # run every 50 ms
            ]
        ),

        Node(
            package='visionsystemx',
            executable='yolo',
            name='yolo',
            output='screen',
            parameters=[
                # {'engine_path': '/home/vanttec/vanttec_usv/RB2025v1.engine'},
                {'engine_path': '/home/vanttec/vanttec_usv/RB2024v2.engine'},
                {'video_topic': '/bebblebrox/video'},
                {'output_topic': '/shapes/detections'},
                {'threshold': 0.1},
            ],

            # ros debug printing
            #arguments=['--ros-args', '--log-level', 'DEBUG']

        ),

        Node(
            package="rviz2",
            executable="rviz2"
        )

    ])
