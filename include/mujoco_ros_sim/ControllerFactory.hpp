#pragma once
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include "mujoco_ros_sim/ControllerInterface.hpp"


class ControllerFactory 
{
public:
  virtual ~ControllerFactory() = default;
  virtual std::shared_ptr<ControllerInterface>
  create(const rclcpp::Node::SharedPtr& node) = 0;
};

