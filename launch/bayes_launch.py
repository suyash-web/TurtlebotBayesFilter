"""
bayes_launch.py
----------------
Launches the full Bayes-filter stack:
  1. Stage simulator   (world=cave)
  2. map_server        (static map for RViz background)
  3. BayesFilterNode   (our Histogram Filter)
  4. RViz2             (with the provided config)

Usage:
    ros2 launch ras598_assignment_3 bayes_launch.py
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    pkg_share = get_package_share_directory('ras598_assignment_3')

    # ── Arguments ────────────────────────────────────────
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use simulated (Stage) clock'
    )
    use_sim_time = LaunchConfiguration('use_sim_time')

    # ── Stage simulator ───────────────────────────────────
    stage_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('stage_ros2'), 'launch', 'demo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': 'cave',
            'use_stamped_velocity': 'false',
        }.items(),
    )

    # ── map_server (serves static occupancy grid for RViz) ─
    map_yaml = os.path.join(pkg_share, 'config', 'cave_map.yaml')
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[
            {'yaml_filename': map_yaml},
            {'use_sim_time': use_sim_time},
        ],
    )

    # Lifecycle manager to bring map_server to ACTIVE
    lifecycle_mgr = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_map',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'autostart': True},
            {'node_names': ['map_server']},
        ],
    )

    # ── Bayes Filter node ─────────────────────────────────
    bayes_node = Node(
        package='ras598_assignment_3',
        executable='bayes_filter',
        name='bayes_filter',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # ── Static TF: odom -> map  (identity – Stage provides odom) ──
    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_odom_tf',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
    )

    # ── RViz ─────────────────────────────────────────────
    rviz_config = os.path.join(pkg_share, 'config', 'bayes.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    return LaunchDescription([
        use_sim_time_arg,
        stage_launch,
        static_tf_node,
        TimerAction(period=2.0, actions=[map_server_node]),
        TimerAction(period=2.0, actions=[lifecycle_mgr]),
        TimerAction(period=3.0, actions=[bayes_node]),
        TimerAction(period=4.0, actions=[rviz_node]),
    ])
