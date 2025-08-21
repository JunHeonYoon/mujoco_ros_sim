cmake_minimum_required(VERSION 3.16)     # cmake ver
project(mujoco_ros_sim)                  # project name

if(NOT CMAKE_BUILD_TYPE)                 # default build
  set(CMAKE_BUILD_TYPE Release )
endif()

# --- Dependencies (C++) ---
find_package(ament_cmake REQUIRED)       # ament
find_package(rclcpp REQUIRED)            # rclcpp
find_package(pluginlib REQUIRED)         # pluginlib
find_package(Eigen3 REQUIRED)            # eigen
find_package(OpenCV REQUIRED)            # opencv
find_package(cv_bridge REQUIRED)         # cv_bridge
find_package(Python3 REQUIRED COMPONENTS Development)  # python dev
find_package(mujoco_ros_sim_msgs REQUIRED)             # msgs

find_package(pybind11 QUIET CONFIG)      # pybind11 (config)

# 2) If not found, try ROS vendor
if(NOT pybind11_FOUND)                   # fallback vendor
  find_package(pybind11_vendor QUIET)
  if(pybind11_vendor_FOUND)
    find_package(pybind11 REQUIRED CONFIG)  # provided by vendor
  endif()
endif()

# 3) We always need the Python dev library for embedding fallback
find_package(Python3 REQUIRED COMPONENTS Development)  # python dev (again)

# Choose link target: prefer pybind11::embed, else Python3::Python
set(PYBIND_LINK_TARGET Python3::Python)  # default link
if(TARGET pybind11::embed)               # prefer embed
  set(PYBIND_LINK_TARGET pybind11::embed)
endif()

# --- Includes ---
include_directories(                      # include dirs
  include
  ${EIGEN3_INCLUDE_DIRS}
)

# --- Plugin library (PyController adapter) ---
add_library(MujocoControllerPlugins SHARED  # plugin lib
  src/PyController.cpp
)
ament_target_dependencies(MujocoControllerPlugins  # plugin deps
  rclcpp pluginlib sensor_msgs std_msgs mujoco_ros_sim_msgs
)
target_include_directories(MujocoControllerPlugins PRIVATE ${Python3_INCLUDE_DIRS}) # py inc
# pybind11 헤더가 있으면 추가
if(pybind11_FOUND AND DEFINED pybind11_INCLUDE_DIRS)  # pybind inc
  target_include_directories(MujocoControllerPlugins PRIVATE ${pybind11_INCLUDE_DIRS})
endif()

target_link_libraries(MujocoControllerPlugins        # plugin link
  ${PYBIND_LINK_TARGET}
  ${OpenCV_LIBS}
)

# --- Node executable ---
add_executable(MujocoController            # node exe
  src/MujocoController.cpp
)
ament_target_dependencies(MujocoController # node deps
  rclcpp pluginlib cv_bridge sensor_msgs std_msgs mujoco_ros_sim_msgs
)
target_link_libraries(MujocoController     # node link
Python3::Python
  MujocoControllerPlugins
  ${OpenCV_LIBS}
)

# --- Install (C++ targets) ---
install(TARGETS MujocoControllerPlugins    # install lib
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(TARGETS MujocoController           # install exe
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

target_link_options(MujocoController PRIVATE -Wl,-E)  # export symbols

# --- Headers / Plugin XML ---
install(DIRECTORY include/ DESTINATION include)               # headers
install(FILES plugins.xml DESTINATION share/${PROJECT_NAME})  # plugin xml

install(DIRECTORY   launch                                   # launch
        DESTINATION share/${PROJECT_NAME})

install(DIRECTORY   mujoco_menagerie                         # assets
        DESTINATION share/${PROJECT_NAME})

install(FILES       resource/${PROJECT_NAME}                 # ament idx
        DESTINATION share/ament_index/resource_index/packages)

install(FILES       package.xml                              # package.xml
        DESTINATION share/${PROJECT_NAME})

ament_python_install_package(mujoco_ros_sim)                 # py pkg

install(PROGRAMS    ${CMAKE_CURRENT_SOURCE_DIR}/mujoco_ros_sim/mujoco_sim.py  # script
        DESTINATION lib/${PROJECT_NAME}
        RENAME      mujoco_sim)

# --- pluginlib export ---
pluginlib_export_plugin_description_file(mujoco_ros_sim plugins.xml)  # export

# --- Export ---
ament_export_include_directories(include)   # export inc
ament_export_dependencies(                  # export deps
  rclcpp pluginlib Eigen3 OpenCV cv_bridge pybind11_vendor
  sensor_msgs std_msgs mujoco_ros_sim_msgs
)

ament_package()                              # ament end
