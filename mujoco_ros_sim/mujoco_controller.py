#!/usr/bin/env python3
import inspect
import numpy as np
import time

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rcl_interfaces.srv import GetParameters

from mujoco_ros_sim_msgs.msg import JointDict, SensorDict, CtrlDict, NamedFloat64Array
from mujoco_ros_sim_msgs.msg import ImageDict

from mujoco_ros_sim.utils import load_class, from_NamedFloat64ArrayMsg, to_NamedFloat64ArrayMsg, image_to_numpy

class MujocoControllerNode(Node):
    def __init__(self):
        super().__init__('mujoco_controller_node')
        
        
        # Parameters/Qos    
        desc = ParameterDescriptor(dynamic_typing=True)
        self.declare_parameter('controller_class', descriptor=desc)
        
        controller_class_str = self.get_parameter('controller_class').get_parameter_value().string_value
            
        qos = QoSProfile(reliability = ReliabilityPolicy.BEST_EFFORT,
                         history     = HistoryPolicy.KEEP_LAST,
                         depth       = 1,
                         durability  = DurabilityPolicy.VOLATILE)
        
        # Controller init
        Controller = load_class(controller_class_str)
        if Controller is None:
            raise RuntimeError("Controller class not found")
        self.controller = Controller(self) if inspect.isclass(Controller) else Controller()
        self.dt_ctrl = float(self.controller.getCtrlTimeStep())
        self.get_logger().info(f"Controller loaded: {controller_class_str} (dt={self.dt_ctrl}s)")
        
        # ROS2 publisher & subscriber
        self.joint_dict_sub   = self.create_subscription(JointDict,  'mujoco_ros_sim/joint_dict',  self.sub_joint_cb,  qos)
        self.sensor_dict_sub  = self.create_subscription(SensorDict, 'mujoco_ros_sim/sensor_dict', self.sub_sensor_cb, qos)
        self.image_dict_sub   = self.create_subscription(ImageDict,  'mujoco_ros_sim/image_dict',  self.sub_image_cb,  qos)
        self.ctrl_command_pub = self.create_publisher(CtrlDict, 'mujoco_ros_sim/ctrl_dict', qos)

        # State cache
        self.latest_joint = None
        self.latest_sensor = None
        self.latest_sim_time = None
        self.started = False

        # Timers loops
        self.create_timer(self.dt_ctrl, self.control_loop)

    def sub_joint_cb(self, msg: JointDict):
        pos = {it.name: from_NamedFloat64ArrayMsg(it) for it in msg.positions}
        vel = {it.name: from_NamedFloat64ArrayMsg(it) for it in msg.velocities}
        tau = {it.name: from_NamedFloat64ArrayMsg(it) for it in msg.torques}
        self.latest_joint = (pos, vel, tau)
        self.latest_sim_time = rclpy.time.Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        
    def sub_sensor_cb(self, msg: SensorDict):
        sen = {it.name: from_NamedFloat64ArrayMsg(it) for it in msg.sensors}
        self.latest_sensor = sen

    def sub_image_cb(self, msg: ImageDict):
        try:
            imgs = {ni.name: image_to_numpy(ni.image) for ni in msg.images}
            self.controller.updateRGBDImage(imgs)
        except Exception as e:
            self.get_logger().warn(f"sub_image_cb error: {e}")

    def control_loop(self):
        if self.latest_joint is None or self.latest_sensor is None:
            return

        t0 = time.perf_counter()
        
        pos, vel, tau, sen = self.latest_joint + (self.latest_sensor or {},)
        sim_time = self.latest_sim_time
        t1 = time.perf_counter()

        self.controller.updateState(pos, vel, tau, sen, sim_time)
        
        if not self.started:
            self.controller.starting()
            self.started = True
        
        t2 = time.perf_counter()
        
        self.controller.compute()
        t3 = time.perf_counter()
        
        ctrl_dict = self.controller.getCtrlInput()  # { actuator_name: float }
        msg = CtrlDict()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.commands = [to_NamedFloat64ArrayMsg(name, val) for name, val in ctrl_dict.items()]
        self.ctrl_command_pub.publish(msg)
        t4 = time.perf_counter()
        
        durations = {
            "getData":     (t1 - t0)*1000,
            "updateState": (t2 - t1)*1000,
            "compute":     (t3 - t2)*1000,
            "applyCtrl":   (t4 - t3)*1000,
            "totalStep":   (t4 - t0)*1000
        }
        
        if self.dt_ctrl - (time.perf_counter() - t0) < 0:
            lines = [
                "\n===================================",
                f"getData took {durations['getData']:.6f} ms",
                f"updateState took {durations['updateState']:.6f} ms",
                f"compute took {durations['compute']:.6f} ms",
                f"applyCtrl took {durations['applyCtrl']:.6f} ms",
                f"totalStep took {durations['totalStep']:.6f} ms",
                "===================================",
            ]
            self.get_logger().warn("\n".join(lines))
        


def main(args=None):
    rclpy.init(args=args)
    node = MujocoControllerNode()
    try:
            rclpy.spin(node)
    except KeyboardInterrupt:
            pass
    finally:
            node.destroy_node()
            rclpy.shutdown()

if __name__ == "__main__":
    main()
