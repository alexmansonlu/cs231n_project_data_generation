import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PythonExpression,PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Find the package where the world and model files are located
    pkg_data_generation = FindPackageShare('data_generation').find('data_generation')
    pkg_gazebo_ros = FindPackageShare(package='gazebo_ros').find('gazebo_ros')   
    # Define the path to the custom world file
    #world_path = os.path.join(pkg_data_generation, 'Worlds', 'metrics.world')
    worlds_folder = os.path.join(pkg_data_generation, 'Worlds')

    # Set the GAZEBO_MODEL_PATH environment variable for custom models (if any)
    #gazebo_models_path = os.path.join(pkg_data_generation, 'models')
    gazebo_models_path = "/home/alexmanson/Documents/stanford/dg_ws/src/data_generation/models"
    gazebo_models_path2 = "/home/alexmanson/Documents/stanford/gazebo_ycb/models"

    
    combined_model_path = f"{gazebo_models_path}:{gazebo_models_path2}"
    os.environ["GAZEBO_MODEL_PATH"] = combined_model_path
    set_model_path = SetEnvironmentVariable(
        name='GAZEBO_MODEL_PATH',
        value=combined_model_path
    )


    # Print paths for debugging
    # print("World path:", world_path)
    # print("Model path:", gazebo_models_path)
    # print("Current GAZEBO_MODEL_PATH:", os.getenv("GAZEBO_MODEL_PATH"))


    ########### YOU DO NOT NEED TO CHANGE ANYTHING BELOW THIS LINE ##############  
    # Launch configuration variables specific to simulation
    headless = LaunchConfiguration('headless')
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_simulator = LaunchConfiguration('use_simulator')
    world_name = LaunchConfiguration('world_name')
    
    declare_simulator_cmd = DeclareLaunchArgument(
        name='headless',
        default_value='False',
        description='Whether to execute gzclient')
        
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        name='use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true')
    
    declare_use_simulator_cmd = DeclareLaunchArgument(
        name='use_simulator',
        default_value='True',
        description='Whether to start the simulator')
    
    declare_world_cmd = DeclareLaunchArgument(
        name='world_name',
        default_value='metrics.world',
        description='Full path to the world model file to load')
        
    # Specify the actions
    print("World name:", world_name)
    
    # Start Gazebo server
    start_gazebo_server_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')),
        condition=IfCondition(use_simulator),
        launch_arguments={'world': PathJoinSubstitution([
            worlds_folder, world_name
        ])}.items()
    )
    
    # Start Gazebo client    
    start_gazebo_client_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')),
        condition=IfCondition(PythonExpression([use_simulator, ' and not ', headless])))
    
    # Create the launch description and populate
    ld = LaunchDescription()
    
    # Declare the launch options
    ld.add_action(declare_simulator_cmd)
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_use_simulator_cmd)
    ld.add_action(declare_world_cmd)
    
    # Add any actions
    ld.add_action(start_gazebo_server_cmd)
    ld.add_action(start_gazebo_client_cmd)
    
    return ld
