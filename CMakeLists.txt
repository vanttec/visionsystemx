cmake_minimum_required(VERSION 3.8)

include(CMakePrintHelpers)

set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(visionsystemx LANGUAGES C CXX CUDA)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(usv_interfaces REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA 10.2 REQUIRED)
find_package(ZED 4 REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(message_filters REQUIRED)
find_package(Eigen3 REQUIRED)

set(TensorRT_INCLUDE_DIRS /usr/include/aarch64-linux-gnu)
set(TensorRT_LIBRARIES /usr/lib/aarch64-linux-gnu)

add_executable(beeblebrox src/beeblebrox.cpp)

ament_target_dependencies(beeblebrox
  rclcpp std_msgs OpenCV cv_bridge image_transport CUDA ZED PCL pcl_conversions usv_interfaces
)
target_include_directories(beeblebrox PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(beeblebrox ${CUDA_LIBRARIES} ${TensorRT_LIBRARIES} nvinfer nvinfer_plugin)

add_executable(yolo src/detectors/yolo_tensorrt/yolo_tensorrt_node.cpp)
ament_target_dependencies(yolo
  rclcpp std_msgs OpenCV cv_bridge image_transport CUDA ZED usv_interfaces
)
target_include_directories(yolo PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(yolo ${CUDA_LIBRARIES} ${TensorRT_LIBRARIES} nvinfer nvinfer_plugin)

add_executable(lidar src/lidar/translation.cpp)
ament_target_dependencies(lidar
  rclcpp std_msgs OpenCV cv_bridge image_transport CUDA ZED PCL pcl_conversions usv_interfaces message_filters Eigen3
)
target_include_directories(lidar PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(lidar ${CUDA_LIBRARIES} ${TensorRT_LIBRARIES} nvinfer nvinfer_plugin ${Eigen3_LIBRARIES})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# install python scripts
file(GLOB SCRIPTS "scripts/*.py")
file(GLOB_RECURSE PY_DETECTORS "src/detectors/*.py")
install(PROGRAMS ${SCRIPTS} ${PY_DETECTORS} DESTINATION lib/${PROJECT_NAME})

install(TARGETS
  beeblebrox
  yolo
  lidar
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  # data
  config
  DESTINATION share/${PROJECT_NAME}/
)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/data)
  install(DIRECTORY data DESTINATION share/${PROJECT_NAME}/)
endif()

ament_package()
