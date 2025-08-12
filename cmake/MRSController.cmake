function(mrs_add_controller TARGET)
  if(ARGN STREQUAL "")
    message(FATAL_ERROR "mrs_add_controller(${TARGET}) needs at least one source file")
  endif()

  add_library(${TARGET} SHARED ${ARGN})
  set_target_properties(${TARGET} PROPERTIES PREFIX "" OUTPUT_NAME ${TARGET})

  find_package(Eigen3 REQUIRED)
  find_package(rclcpp  REQUIRED)
  find_package(eigenpy REQUIRED)
  find_package(OpenCV REQUIRED COMPONENTS core imgproc)
  find_package(cv_bridge REQUIRED)
  find_package(Boost REQUIRED COMPONENTS
    python${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}
    numpy${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}
  )

  target_link_libraries(${TARGET} PUBLIC
    mujoco_ros_sim::bindings  # registry + bridge
    rclcpp::rclcpp
    Eigen3::Eigen
    eigenpy::eigenpy
    ${OpenCV_LIBS}
    )

  ament_target_dependencies(${TARGET} PUBLIC cv_bridge)

  target_include_directories(${TARGET}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:include>
           ${OpenCV_INCLUDE_DIRS})

  install(TARGETS ${TARGET}
    LIBRARY DESTINATION lib/${PROJECT_NAME}_plugins)
endfunction()
