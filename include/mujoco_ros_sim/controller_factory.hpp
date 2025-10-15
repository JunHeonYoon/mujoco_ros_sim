#pragma once
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include "mujoco_ros_sim/controller_interface.hpp"

namespace MujocoRosSim
{
  class ControllerFactory 
  {
  public:
    virtual ~ControllerFactory() = default;
    virtual std::shared_ptr<ControllerInterface>
    create(const rclcpp::Node::SharedPtr& node) = 0;
  };
} // namespace MujocoRosSim

