from rclpy.node import Node
from typing import Dict, Any


class ControllerInterface:
    def __init__(self, node: Node):
        self.node = node
        self.dt   = 0.001  # Default time step, can be overridden by subclasses

    def starting(self) -> None:
        raise NotImplementedError

    def updateState(
        self,
        pos_dict: Dict[str, Any],
        vel_dict: Dict[str, Any],
        tau_ext_dict: Dict[str, Any],
        sensor_dict: Dict[str, Any],
        current_time: float
    ) -> None:
        raise NotImplementedError
    
    def updateRGBDImage(self, rgbd_dict: Dict[str, Any]) ->None:
        pass
        
    def compute(self) -> None:
        raise NotImplementedError

    def getCtrlInput(self) -> Dict[str, float]:
        raise NotImplementedError
    
    def getCtrlTimeStep(self) -> float:
        return float(self.dt)
