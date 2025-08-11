#pragma once

#include <rclcpp/rclcpp.hpp>
#include <unordered_map>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include "mujoco_ros_sim/ControllerRegistry.hpp"

using Vec          = Eigen::VectorXd;
using VecMap       = std::unordered_map<std::string, Vec>;
using VecMap       = std::unordered_map<std::string, Vec>;
using CtrlInputMap = std::unordered_map<std::string, double>;
using ImageCVMap   = std::unordered_map<std::string, cv::Mat>;

/**
 * @brief Minimal controller interface for mujoco_ros_sim.
 *
 * Tick order per control cycle:
 *   1) updateState(...)
 *   2) starting()            // first tick only
 *   3) compute()
 *   4) getCtrlInput()
 *
 * Implement in derived classes: starting(), updateState(), compute(), getCtrlInput().
 * Optional to override: updateRGBDImage().
*/
class ControllerInterface
{
public:
  /**
   * @brief Construct with a ROS 2 node handle.
   *
   * @param node (rclcpp::Node::SharedPtr) Node to access parameters, pubs/subs, services.
   *
   * What you do here (in your derived constructor):
   * - Optionally declare/read parameters and set dt_ if you need a custom control period.
   * - Create publishers/subscriptions if your controller needs them.
  */
  ControllerInterface(const rclcpp::Node::SharedPtr& node)
    : node_(node)
  {
    exec_.add_node(node_);
    spin_thread_ = std::thread([this]{
      rclcpp::Rate rate(5000.0);
      while (running_) { exec_.spin_some(); rate.sleep(); }
    });
  }

  virtual ~ControllerInterface() = default;

  /**
   * @brief One-time initialization hook (first control tick only).
   *
   * What to do:
   * - Allocate buffers, precompute constants, read/validate parameters.
   * - Initialize estimators/filters/integrators.
  */
  virtual void starting() = 0;

  /**
   * @brief Provide the latest simulator state.
   *
   * @param pos      (VecMap) joint name (std::string) -> joint position (Eigen::VectorXd) map.
   *                 Each vector size matches that joint’s DOF in qpos.
   * @param vel      (VecMap) joint name (std::string) -> joint velocity (Eigen::VectorXd) map.
   *                 Each vector size matches that joint’s DOF in qvel.
   * @param tau_ext  (VecMap) joint name (std::string) -> joint effort/torque (Eigen::VectorXd) map
   *                 as published by the simulator integration (e.g., actuator effort).
   * @param sensors  (VecMap) sensor name (std::string) -> raw sensor vector (Eigen::VectorXd) map
   *                 sliced from sensordata.
   * @param sim_time (double) simulation time in seconds.
   *
   * What to do:
   * - Copy/consume only what you need into your internal state (keep this fast).
   * - Do NOT block or perform heavy optimization here; defer to compute().
   * - If you share state with other callbacks (e.g., images), guard with a mutex.
  */
  virtual void updateState(const VecMap& pos,
                           const VecMap& vel,
                           const VecMap& tau_ext,
                           const VecMap& sensors,
                           double sim_time) = 0;

  /**
   * @brief Optional RGB-D/image update (runs ~cam_fps from /mujoco_sim_node/camera_fps).
   *
   * @param rgbd (ImageCVMap) image stream name (std::string) -> OpenCV image
   *             (cv::Mat). 
   *
   * What to do:
   * - Perform lightweight preprocessing or store references/copies for use in compute().
   * - It’s fine to leave this unimplemented if your controller doesn’t use images.
  */
  virtual void updateRGBDImage(const ImageCVMap& images) {}

  /**
   * @brief Compute control outputs using the most recent inputs.
   *
   * What to do:
   * - Implement your control law (e.g., PID/impedance/MPC/etc.).
   * - Use the state cached in updateState()/updateRGBDImage().
   * - Store results internally so getCtrlInput() can return them immediately.
   * - Keep this bounded in time; it runs every control tick.
  */
  virtual void compute() = 0;

  /**
   * @brief Retrieve actuator commands computed by compute().
   *
   * @return (CtrlInputMap) actuator name (std::string) -> command value (double).
   *
   * What to do:
   * - Return finite, valid commands for all actuators you control.
   * - Do not allocate excessively; prefer returning a cached map updated in compute().
   * - Units should match the simulator’s actuator model (e.g., torque/position/velocity).
  */
  virtual CtrlInputMap getCtrlInput() const = 0;

  /**
   * @brief Desired control period used by the runner.
   *
   * @return (double) control timestep in seconds. Default: 0.001 (1 kHz).
   *
   * What to do:
   * - If you need a different rate, set dt_ (e.g., in your constructor or starting()).
   * - Choose a period your compute() can reliably meet on your target CPU.
  */
  const double getCtrlTimeStep() const { return dt_; }

protected:
  rclcpp::Node::SharedPtr node_;       // ROS 2 node handle
  double                  dt_{0.001};  // Control period [s]

private:
  // Internal: ROS executor thread for this node (no action required by users).
  rclcpp::executors::SingleThreadedExecutor exec_;
  std::thread                               spin_thread_;
  std::atomic_bool                          running_{true};
};
