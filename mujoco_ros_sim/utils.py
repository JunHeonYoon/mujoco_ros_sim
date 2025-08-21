import mujoco
import os
from ament_index_python.packages import get_package_share_directory
import time
import importlib
import numpy as np
import math
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, MultiArrayLayout
from sensor_msgs.msg import Image
from builtin_interfaces.msg import Time
from mujoco_ros_sim_msgs.msg import NamedFloat64Array

RED   = "\033[1;31m"
YEL   = "\033[1;33m"
GRN   = "\033[1;32m"
BLU   = "\033[1;34m"
RESET = "\033[0m"

def load_mj_model(robot_name) -> mujoco.MjModel:
    """
    Loads a MuJoCo model from an XML file based on the provided robot name.

    This function constructs a path to the "mujoco_menagerie" directory within the package's share directory,
    checks if the specified robot exists, and then loads the corresponding MuJoCo model using its XML description.

    Parameters:
        robot_name (str): The name of the robot to load. This should match one of the subdirectories 
                          in the "mujoco_menagerie" directory.

    Returns:
        mujoco.MjModel: The MuJoCo model object loaded from the XML file corresponding to the given robot name.

    Raises:
        AssertionError: If the specified robot_name is not found in the available robot directories.
    """
    # Construct the path to the "mujoco_menagerie" directory in the package's share directory.
    mujoco_menagerie_path = os.path.join(get_package_share_directory(__package__), "mujoco_menagerie")
    
    # List all available robot directories in the mujoco_menagerie folder.
    available_robots = [name for name in os.listdir(mujoco_menagerie_path)
                        if os.path.isdir(os.path.join(mujoco_menagerie_path, name))]
    
    # Assert that the provided robot_name is among the available robots.
    assert robot_name in available_robots, f"{robot_name} is not included in {available_robots}!"
    
    # Build the full path to the XML file that describes the MuJoCo scene for the robot.
    xml_file_path = get_package_share_directory(__package__) + f"/mujoco_menagerie/{robot_name}/scene.xml"
    
    # Load and return the MuJoCo model from the XML file.
    return mujoco.MjModel.from_xml_path(xml_file_path)

def precise_sleep(duration):
    """
    Sleeps for a specified duration using a busy-wait loop with a high-resolution timer.

    This function uses a while-loop along with a high-resolution performance counter (perf_counter)
    to pause execution for the given duration. This approach is intended for use cases where
    precise timing is required, such as in simulation loops.

    Parameters:
        duration (float): The amount of time in seconds to sleep.

    Returns:
        None
    """
    # Record the start time using a high-resolution performance counter.
    start = time.perf_counter()
    
    # Busy-wait loop until the elapsed time reaches or exceeds the specified duration.
    while True:
        now = time.perf_counter()
        if (now - start) >= duration:
            break

def load_class(full_class_string: str):
    """
    Dynamically loads a class from a full class path string.

    This function handles cases where the input is None or an empty string,
    and verifies that the string contains at least one dot ('.') to separate
    the module and class names. If the input is valid, it imports the module and
    retrieves the class.

    Parameters:
        full_class_string (str): A string representing the full path of the class,
                                 in the format "package.module.ClassName". For example,
                                 "my_package.my_module.MyClass".

    Returns:
        type or None: The class specified by the input string if found; otherwise,
                      None if the input is None or an empty string.

    Raises:
        ValueError: If the input string does not contain a dot ('.'), indicating
                    that it is not in the expected "module.ClassName" format.
    """
    # Check if the provided string is None or empty.
    if not full_class_string:
        # Return None if there is no valid class string.
        return None
    # Ensure that the string includes a dot to separate the module and class names.
    if '.' not in full_class_string:
        # Raise an error if the format is incorrect.
        raise ValueError(f"Invalid class string: '{full_class_string}'. "
                         "Must be in the form 'package.module.ClassName'.")

    # Split the full class string into the module name and the class name.
    # rsplit is used with maxsplit=1 to split only at the last occurrence of '.'
    module_name, class_name = full_class_string.rsplit('.', 1)
    
    # Dynamically import the module using the module name.
    mod = importlib.import_module(module_name)
    
    # Retrieve the class attribute from the module using the class name.
    cls = getattr(mod, class_name)
    
    # Return the loaded class.
    return cls

def print_table(robot_name: str, m: mujoco.MjModel) -> str:
    jt_enum = mujoco.mjtJoint
    enum2name = {getattr(jt_enum, a): a[5:].title() for a in dir(jt_enum) if a.startswith("mjJNT_")}

    lines = []
    lines.append("=================================================================")
    lines.append("=================================================================")
    lines.append(f"MuJoCo Model Information: {robot_name}")
    lines.append(" id | name                 | type   | nq | nv | idx_q | idx_v")
    lines.append("----+----------------------+--------+----+----+-------+------")

    for jid in range(m.njnt):
        adr  = m.name_jntadr[jid]
        name = m.names[adr:].split(b'\x00', 1)[0].decode()
        if not name:
            continue
        jtype    = int(m.jnt_type[jid])
        type_str = enum2name.get(jtype, "Unk")
        idx_q = int(m.jnt_qposadr[jid])
        idx_v = int(m.jnt_dofadr[jid])
        next_q = m.jnt_qposadr[jid + 1] if jid + 1 < m.njnt else m.nq
        next_v = m.jnt_dofadr[jid + 1] if jid + 1 < m.njnt else m.nv
        nq = int(next_q - idx_q)
        nv = int(next_v - idx_v)
        lines.append(f"{jid:3d} | {name:20s} | {type_str:6s} | {nq:2d} | {nv:2d} | {idx_q:5d} | {idx_v:4d}")

    lines.append("")
    trn_enum = mujoco.mjtTrn
    trn2name = {getattr(trn_enum, a): a[5:].title() for a in dir(trn_enum) if a.startswith("mjTRN_")}
    joint_names = {jid: m.names[m.name_jntadr[jid]:].split(b'\x00', 1)[0].decode() for jid in range(m.njnt)}
    lines.append(" id | name                 | trn     | target_joint")
    lines.append("----+----------------------+---------+-------------")
    for aid in range(m.nu):
        adr  = m.name_actuatoradr[aid]
        name = m.names[adr:].split(b'\x00', 1)[0].decode()
        trn_type = int(m.actuator_trntype[aid])
        trn_str  = trn2name.get(trn_type, "Unk")
        target_joint = "-"
        if trn_type in (trn_enum.mjTRN_JOINT, trn_enum.mjTRN_JOINTINPARENT):
            j_id = int(m.actuator_trnid[aid, 0])
            target_joint = joint_names.get(j_id, str(j_id))
        lines.append(f"{aid:3d} | {name:20s} | {trn_str:7s} | {target_joint}")

    lines.append("")
    sens_enum = mujoco.mjtSensor
    sens2name = {getattr(sens_enum, a): a[7:].title() for a in dir(sens_enum) if a.startswith("mjSENS_")}
    obj_enum  = mujoco.mjtObj
    obj2name  = {getattr(obj_enum, a): a[6:].title() for a in dir(obj_enum) if a.startswith("mjOBJ_")}
    body_names = {bid: m.names[m.name_bodyadr[bid]:].split(b'\0', 1)[0].decode() for bid in range(m.nbody)}
    site_names = {sid: m.names[m.name_siteadr[sid]:].split(b'\0', 1)[0].decode() for sid in range(m.nsite)}

    def obj_name(objtype, objid):
        if objtype == obj_enum.mjOBJ_BODY:
            return body_names.get(objid, str(objid))
        if objtype == obj_enum.mjOBJ_SITE:
            return site_names.get(objid, str(objid))
        if objtype == obj_enum.mjOBJ_JOINT:
            return joint_names.get(objid, str(objid))
        return str(objid)

    lines.append(" id | name                        | type             | dim | adr | target (obj)")
    lines.append("----+-----------------------------+------------------+-----+-----+----------------")
    for sid in range(m.nsensor):
        adr  = m.name_sensoradr[sid]
        name = m.names[adr:].split(b'\0', 1)[0].decode()
        stype = int(m.sensor_type[sid])
        tstr  = sens2name.get(stype, "Unk")
        dim   = int(m.sensor_dim[sid])
        sadr  = int(m.sensor_adr[sid])
        objtype = int(m.sensor_objtype[sid])
        objid   = int(m.sensor_objid[sid])
        target  = f"{obj2name.get(objtype,'-')}:{obj_name(objtype,objid)}" if objid >= 0 else "-"
        lines.append(f"{sid:3d} | {name:27s} | {tstr:16s} | {dim:3d} | {sadr:3d} | {target}")

    lines.append("")
    lines.append(" id | name                        | mode     | resolution")
    lines.append("----+-----------------------------+----------+------------")
    mode_map = {}
    cam_enum = getattr(mujoco, "mjtCamLight", None)
    if cam_enum is not None:
        for attr in dir(cam_enum):
            if attr.startswith("mjCAMLIGHT_"):
                mode_map[getattr(cam_enum, attr)] = attr[10:].title()
    else:
        mode_map = {0: "Fixed", 1: "Track", 2: "Trackcom"}

    try:
        offw = int(m.vis.global_.offwidth)
        offh = int(m.vis.global_.offheight)
        res_str = f"{offw}x{offh}" if offw > 0 and offh > 0 else "-"
    except Exception:
        res_str = "-"

    ncam = getattr(m, "ncam", 0)
    for cid in range(ncam):
        cadr  = m.name_camadr[cid]
        cname = m.names[cadr:].split(b'\x00', 1)[0].decode()
        if hasattr(m, "cam_mode"):
            try:
                mode_val = int(m.cam_mode[cid])
                mode_str = mode_map.get(mode_val, str(mode_val))
            except Exception:
                mode_str = "-"
        else:
            mode_str = "-"
        lines.append(f"{cid:3d} | {cname:27s} | {mode_str:8s} | {res_str}")
    lines.append("=================================================================")
    lines.append("=================================================================")
    return "\n".join(lines)

def to_NamedFloat64ArrayMsg(name: str, data):
    arr = np.array(data, dtype=float).ravel() # flatten to 1D array
    msg = NamedFloat64Array()
    msg.name = name
    fa = Float64MultiArray()
    fa.layout = MultiArrayLayout(dim=[MultiArrayDimension(label="", size=arr.size, stride=arr.size)], data_offset=0)
    fa.data = arr.tolist()
    msg.value = fa
    return msg
    
def to_TimeMsg(time: float):
    time_msg = Time()
    time_msg.sec = int(math.floor(time))
    time_msg.nanosec = int(round((time - time_msg.sec) * 1e9))
    return time_msg

def from_NamedFloat64ArrayMsg(item: NamedFloat64Array) -> np.ndarray:
    a = np.array(item.value.data, dtype=float)
    return a if a.size != 1 else a

def image_to_numpy(img_msg: Image) -> np.ndarray:
    h, w = int(img_msg.height), int(img_msg.width)
    enc = img_msg.encoding.lower()
    buf = memoryview(img_msg.data)
    if enc in ("rgb8", "bgr8"):
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
        if enc == "bgr8":
            arr = arr[..., ::-1]
        return arr
    elif enc in ("mono8",):
        return np.frombuffer(buf, dtype=np.uint8).reshape(h, w)
    elif enc in ("32fc1",):
        return np.frombuffer(buf, dtype=np.float32).reshape(h, w)
    elif enc in ("32fc3",):
        return np.frombuffer(buf, dtype=np.float32).reshape(h, w, 3)
    else:
        return np.frombuffer(buf, dtype=np.uint8)


class PreciseSleeper:
    def __init__(self, sample=1e-3, pessimism=1.0):
        self.sample = sample
        self.pess   = pessimism
        self.n = 1
        self.mean = 5e-3
        self.m2 = 0.0
        self.est = self.mean

    def _upd(self, obs):
        self.n += 1
        d = obs - self.mean
        self.mean += d / self.n
        self.m2   += d * (obs - self.mean)
        if self.n > 1:
            std = math.sqrt(self.m2 / (self.n - 1))
            self.est = self.mean + self.pess * std
        else:
            self.est = self.mean

    @staticmethod
    def _spin(sec):
        if sec <= 0: return
        end = time.perf_counter_ns() + int(sec * 1e9)
        while time.perf_counter_ns() < end:
            pass

    def coarse(self, remain):
        """remain 초 남았을 때, oversleep을 피하며 coarse sleep."""
        while remain - self.est > 1e-6:
            t0 = time.perf_counter_ns()
            time.sleep(self.sample)
            obs = (time.perf_counter_ns() - t0)/1e9
            remain -= obs
            self._upd(obs)
        return remain  # 남은 꼬리는 spin

class AccuratePeriodic:
    """절대 데드라인 기반 + overshoot 예측"""
    def __init__(self, period_s: float):
        self.period_ns = int(period_s * 1e9)
        self.deadline  = time.perf_counter_ns() + self.period_ns
        self.sleeper   = PreciseSleeper(sample=1e-3, pessimism=1.0)

        # 스케줄링 오버헤드(조기 기상 오차) EWMA 추정치 (us 단위)
        self.bias_us = 0.0
        self.alpha = 0.1  # EWMA 계수

    def wait_next(self) -> float:
        # oversleep/오버헤드 보정만큼 "조금 일찍" 깨우기
        target_ns = self.deadline - int(max(0.0, self.bias_us) * 1e3)

        while True:
            now = time.perf_counter_ns()
            remain_ns = target_ns - now
            if remain_ns <= 0:
                break
            remain = remain_ns / 1e9
            remain = self.sleeper.coarse(remain)  # sleep(1ms) 반복
            self.sleeper._spin(remain)            # 꼬리 spin


        wake = time.perf_counter_ns()
        # 실제 데드라인 기준 늦잠(+): us
        lateness_us = (wake - self.deadline) / 1e3

        # 다음 데드라인으로 전진 (드리프트 없음)
        self.deadline += self.period_ns
        if wake > self.deadline:
            miss = (wake - self.deadline) // self.period_ns + 1
            self.deadline += miss * self.period_ns

        # bias 업데이트: "얼마나 일찍 깨워야 제시간이 되는가"를 학습
        self.bias_us = (1 - self.alpha) * self.bias_us + self.alpha * max(0.0, lateness_us)
        return lateness_us
