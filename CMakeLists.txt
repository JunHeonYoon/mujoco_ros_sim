cmake_minimum_required(VERSION 3.16)     # cmake ver
project(mujoco_ros_sim)                  # project name

if(NOT CMAKE_BUILD_TYPE)                 # default build
  set(CMAKE_BUILD_TYPE Release )
endif()

set(CMAKE_POLICY_DEFAULT_CMP0072 NEW)

if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif ()

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


# =====================================================================
# download dependencies
include(FetchContent)

set(MUJOCO_DEP_VERSION_lodepng
    b4ed2cd7ecf61d29076169b49199371456d4f90b
    CACHE STRING "Version of `lodepng` to be fetched."
)

# Fetch lodepng dependency.
if (NOT TARGET lodepng)
  FetchContent_Declare(
      lodepng
      GIT_REPOSITORY https://github.com/lvandeve/lodepng.git
      GIT_TAG ${MUJOCO_DEP_VERSION_lodepng}
  )

  FetchContent_GetProperties(lodepng)
  if (NOT lodepng_POPULATED)
    FetchContent_Populate(lodepng)
    # This is not a CMake project.
    set(LODEPNG_SRCS ${lodepng_SOURCE_DIR}/lodepng.cpp)
    set(LODEPNG_HEADERS ${lodepng_SOURCE_DIR}/lodepng.h)
    add_library(lodepng STATIC ${LODEPNG_HEADERS} ${LODEPNG_SRCS})
    target_compile_options(lodepng PRIVATE ${MUJOCO_MACOS_COMPILE_OPTIONS})
    target_link_options(lodepng PRIVATE ${MUJOCO_MACOS_LINK_OPTIONS})
    target_include_directories(lodepng PUBLIC ${lodepng_SOURCE_DIR})
  endif ()
endif ()

# find dependencies
set(OpenGL_GL_PREFERENCE GLVND)
# find_package(ament_cmake REQUIRED)
find_package(mujoco REQUIRED)
find_package(glfw3 REQUIRED)
find_package(Threads REQUIRED)

# find_package(rclcpp REQUIRED)
# find_package(pluginlib REQUIRED)
# find_package(controller_manager REQUIRED)
find_package(ament_index_cpp REQUIRED)
# =====================================================================


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
  src/mujoco_ros_sim/py_controller.cpp
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
  src/mujoco_ros_sim/mujoco_controller.cpp
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


# ======================================================================
add_library(MujocoSimIface INTERFACE)
target_include_directories(MujocoSimIface
    INTERFACE
      "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
      "$<INSTALL_INTERFACE:include/mujoco>"
      "$<INSTALL_INTERFACE:include/mujoco_ros_sim>")

add_executable(MujocoSim
    src/mujoco/glfw_adapter.cc
    src/mujoco/glfw_dispatch.cc
    src/mujoco/platform_ui_adapter.cc
    src/mujoco/simulate.cc
    src/mujoco_ros_sim/mujoco_sim.cpp)
target_link_libraries(MujocoSim PUBLIC MujocoSimIface mujoco::mujoco glfw lodepng Threads::Threads)
ament_target_dependencies(MujocoSim PUBLIC rclcpp cv_bridge sensor_msgs std_msgs mujoco_ros_sim_msgs)
target_include_directories(MujocoSim PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include/mujoco>"
    "$<INSTALL_INTERFACE:include/mujoco_ros_sim>" PRIVATE src)
install(TARGETS MujocoSim
    RUNTIME DESTINATION lib/${PROJECT_NAME}
)
# ======================================================================


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

# install(PROGRAMS    ${CMAKE_CURRENT_SOURCE_DIR}/mujoco_ros_sim/mujoco_sim.py  # script
#         DESTINATION lib/${PROJECT_NAME}
#         RENAME      mujoco_sim)

# --- pluginlib export ---
pluginlib_export_plugin_description_file(mujoco_ros_sim plugins.xml)  # export

# --- Export ---
ament_export_include_directories(include)   # export inc
ament_export_dependencies(                  # export deps
  rclcpp pluginlib Eigen3 OpenCV cv_bridge pybind11_vendor
  sensor_msgs std_msgs mujoco_ros_sim_msgs mujoco
)

ament_package()                              # ament end