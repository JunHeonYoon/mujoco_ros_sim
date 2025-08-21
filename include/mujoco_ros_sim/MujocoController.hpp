#pragma once

#include <rclcpp/rclcpp.hpp>
#include <pluginlib/class_loader.hpp>
#include <mutex>
#include <chrono>

#include "mujoco_ros_sim/ControllerInterface.hpp"   // VecMap, ImageCVMap 등
#include "mujoco_ros_sim/ControllerFactory.hpp"

// msgs
#include "mujoco_ros_sim_msgs/msg/joint_dict.hpp"
#include "mujoco_ros_sim_msgs/msg/sensor_dict.hpp"
#include "mujoco_ros_sim_msgs/msg/ctrl_dict.hpp"
#include "mujoco_ros_sim_msgs/msg/named_float64_array.hpp"
#include "mujoco_ros_sim_msgs/msg/image_dict.hpp"

class ControllerNode : public rclcpp::Node
{
public:
  ControllerNode();
  void init_controller();

private:
  // callbacks
  void jointCb(const mujoco_ros_sim_msgs::msg::JointDict::SharedPtr msg);
  void sensorCb(const mujoco_ros_sim_msgs::msg::SensorDict::SharedPtr msg);
  void imageCb(const mujoco_ros_sim_msgs::msg::ImageDict::SharedPtr msg);
  void controlLoop();

  // plugin loader 
  pluginlib::ClassLoader<ControllerFactory> loader_{"mujoco_ros_sim", "ControllerFactory"};
  std::shared_ptr<ControllerInterface> controller_;
  std::string controller_class_;

  rclcpp::CallbackGroup::SharedPtr timer_group_;
  rclcpp::CallbackGroup::SharedPtr sub_group_;
  rclcpp::TimerBase::SharedPtr      timer_;

  // subs/pubs
  rclcpp::Subscription<mujoco_ros_sim_msgs::msg::JointDict>::SharedPtr  joint_sub_;
  rclcpp::Subscription<mujoco_ros_sim_msgs::msg::SensorDict>::SharedPtr sensor_sub_;
  rclcpp::Subscription<mujoco_ros_sim_msgs::msg::ImageDict>::SharedPtr  image_sub_;
  rclcpp::Publisher<mujoco_ros_sim_msgs::msg::CtrlDict>::SharedPtr      ctrl_pub_;

  // state cache
  std::mutex state_mtx_;
  VecMap latest_pos_, latest_vel_, latest_tau_, latest_sensors_;
  double latest_time_{0.0};
  bool have_joint_{false};
  bool have_sensor_{false};

  // timer
  std::chrono::nanoseconds period_ns_{std::chrono::milliseconds(1)};

  // once-start
  bool started_{false};
};
