from rclpy.node import Node
from typing import Dict
import numpy as np


class ControllerInterface:
    def __init__(self, node: Node):
        self.node = node
        self.dt   = 0.001  # Default time step, can be overridden by subclasses

    def starting(self) -> None:
        raise NotImplementedError

    def updateState(
        self,
        pos_dict: Dict[str, np.ndarray],
        vel_dict: Dict[str, np.ndarray],
        tau_ext_dict: Dict[str, np.ndarray],
        sensor_dict: Dict[str, np.ndarray],
        current_time: float
    ) -> None:
        raise NotImplementedError
    
    def updateRGBDImage(self, rgbd_dict: Dict[str, Dict[str, np.ndarray]]) ->None:
        pass
        
    def compute(self) -> None:
        raise NotImplementedError

    def getCtrlInput(self) -> Dict[str, float]:
        raise NotImplementedError
    
    def getCtrlTimeStep(self) -> float:
        return float(self.dt)
