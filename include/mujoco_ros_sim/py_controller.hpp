#pragma once
#include "mujoco_ros_sim/controller_interface.hpp"
#include "mujoco_ros_sim/controller_factory.hpp"
#include <memory>

class PyController final : public ControllerInterface 
{
public:
  explicit PyController(const rclcpp::Node::SharedPtr& node);
  ~PyController() override;

  void starting() override;
  void updateState(const VecMap& pos,
                   const VecMap& vel,
                   const VecMap& tau_ext,
                   const VecMap& sensors,
                   double sim_time) override;
  void updateRGBDImage(const ImageCVMap& images) override;
  void compute() override;
    CtrlInputMap getCtrlInput() const override;
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class PyControllerFactory final : public ControllerFactory 
{
public:
  std::shared_ptr<ControllerInterface>
  create(const rclcpp::Node::SharedPtr& node) override {
    return std::make_shared<PyController>(node);
  }
};
