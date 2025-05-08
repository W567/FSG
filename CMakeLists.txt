cmake_minimum_required(VERSION 3.0.2)
project(fsg)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 14)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/")  # qhull

# sequence matters, or project may be linked to libqhull.so instead of libqhull_r.so
find_package(QuickHull REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(PCL 1.8 REQUIRED)
find_package(CGAL REQUIRED Qt5)
find_package(OpenCV REQUIRED )
find_package(OpenMP REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  geometry_msgs
  message_generation
  pcl_ros
  tf
  tf_conversions
  eigen_conversions
  pcl_conversions
  pcl_interface
  sr_description
  robotiq_2f_85_gripper_visualization
  allegro_hand_description
)

add_service_files(
  FILES
  GraspPose.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs geometry_msgs
)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS message_runtime std_msgs geometry_msgs pcl_ros tf tf_conversions
                 eigen_conversions pcl_conversions pcl_interface
)

include_directories(
  include
  ${QHULL_INCLUDE_DIR}
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(fsgServer
    src/main.cc
    src/cgal.cc
    src/eigen.cc
    src/gws.cc
    src/object.cc
    src/visualizer.cc
)
target_link_libraries(fsgServer
  ${QHULL_LIBRARIES}
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  CGAL::CGAL
  OpenMP::OpenMP_CXX
)
add_dependencies(fsgServer fsg_gencpp)

option(BUILD_WITH_VISUALIZER "Build with Visualizer" OFF)
if(BUILD_WITH_VISUALIZER)
  target_compile_definitions(fsgServer PRIVATE BUILD_WITH_VISUALIZER)
endif()
