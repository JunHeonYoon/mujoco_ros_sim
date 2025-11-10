# Mujoco ROS Sim

---
## Dependencies

- [ROS2 Humble](https://docs.ros.org/en/humble/index.html)  
- [MuJoCo](https://mujoco.org/)  
---

## Installation

```bash
sudo apt install ros-$ROS_DISTRO-rmw-cyclonedds-cpp
echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> ~/.bashrc

cd ~/ros2_ws
git clone --recursive https://github.com/JunHeonYoon/mujoco_ros_sim.git src
colcon build --symlink-install
source install/setup.bash
```
