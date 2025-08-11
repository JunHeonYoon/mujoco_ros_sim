import mujoco
import os
from ament_index_python.packages import get_package_share_directory
import time
import importlib
from rclpy.node import Node
import numpy as np
import math
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, MultiArrayLayout
from sensor_msgs.msg import Image
from builtin_interfaces.msg import Time
from mujoco_ros_sim_msgs.msg import NamedFloat64Array

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

def print_table(node: Node, m: mujoco.MjModel):
    # ---------- Helper: build enum-value → name dict for joint types ----------
    jt_enum = mujoco.mjtJoint                 # enum container for joints
    enum2name = {}                            # maps int value → readable str
    for attr in dir(jt_enum):                 # iterate every attribute
        if attr.startswith("mjJNT_"):         # only keep joint type constants
            enum2name[getattr(jt_enum, attr)] = attr[5:].title()  # mjJNT_FREE -> Free

    # ---------- Joint table header ----------
    hdr = " id | name                 | type   | nq | nv | idx_q | idx_v"
    sep = "----+----------------------+--------+----+----+-------+------"
    node.get_logger().info(hdr)
    node.get_logger().info(sep)

    # ---------- Iterate over joints and print per-row ----------
    for jid in range(m.njnt):
        adr  = m.name_jntadr[jid]                         # byte offset in names buffer
        name = m.names[adr:].split(b'\x00', 1)[0].decode()  # null-terminated C-string
        
        if not name:  # if name is empty, use a placeholder
            continue

        jtype     = int(m.jnt_type[jid])                  # numeric enum value
        type_str  = enum2name.get(jtype, "Unk")           # human-readable

        idx_q = int(m.jnt_qposadr[jid])                   # first qpos index
        idx_v = int(m.jnt_dofadr[jid])                    # first qvel (dof) index

        next_q = m.jnt_qposadr[jid + 1] if jid + 1 < m.njnt else m.nq
        next_v = m.jnt_dofadr[jid + 1] if jid + 1 < m.njnt else m.nv

        nq = int(next_q - idx_q)                          # number of qpos entries
        nv = int(next_v - idx_v)                          # number of dof entries

        # Print one formatted line
        node.get_logger().info(
            f"{jid:3d} | {name:20s} | {type_str:6s} |"
            f" {nq:2d} | {nv:2d} | {idx_q:5d} | {idx_v:4d}"
        )

    # ---------- Actuator table (print after a blank line) ----------
    node.get_logger().info("")                            # readability spacer

    # Build enum-value → name dict for actuator transmission types
    trn_enum = mujoco.mjtTrn
    trn2name = {}
    for attr in dir(trn_enum):
        if attr.startswith("mjTRN_"):
            trn2name[getattr(trn_enum, attr)] = attr[5:].title()  # mjTRN_JOINT -> Joint

    # Pre-compute joint ID → name for quick lookup when actuator targets a joint
    joint_names = {
        jid: m.names[m.name_jntadr[jid]:].split(b'\x00', 1)[0].decode()
        for jid in range(m.njnt)
    }

    # Actuator table header
    ahdr = " id | name                 | trn     | target_joint"
    asep = "----+----------------------+---------+-------------"
    node.get_logger().info(ahdr)
    node.get_logger().info(asep)

    # Iterate over all actuators
    for aid in range(m.nu):                                # m.nu == number of actuators
        adr  = m.name_actuatoradr[aid]                     # byte offset to actuator name
        name = m.names[adr:].split(b'\x00', 1)[0].decode()

        trn_type   = int(m.actuator_trntype[aid])          # transmission type enum
        trn_str    = trn2name.get(trn_type, "Unk")         # readable transmission

        # Each actuator has up to two target IDs in actuator_trnid.
        # For JOINT / JOINTINPARENT the first entry is the joint index.
        target_joint = "-"
        if trn_type in (trn_enum.mjTRN_JOINT,
                        trn_enum.mjTRN_JOINTINPARENT):
            j_id = int(m.actuator_trnid[aid, 0])
            target_joint = joint_names.get(j_id, str(j_id))

        # Log actuator info
        node.get_logger().info(
            f"{aid:3d} | {name:20s} | {trn_str:7s} | {target_joint}"
        )
        
    # ---------- Sensor table (print after a blank line) ----------
    node.get_logger().info("")

    # enum → 문자열 매핑 --------------------
    sens_enum = mujoco.mjtSensor
    sens2name = {getattr(sens_enum, a): a[7:].title()
                for a in dir(sens_enum) if a.startswith("mjSENS_")}

    obj_enum  = mujoco.mjtObj
    obj2name  = {getattr(obj_enum, a): a[6:].title()
                for a in dir(obj_enum) if a.startswith("mjOBJ_")}

    # 미리 body / site / joint 등의 id→name 사전 생성
    body_names = {bid: m.names[m.name_bodyadr[bid]:].split(b'\0', 1)[0].decode()
                for bid in range(m.nbody)}
    site_names = {sid: m.names[m.name_siteadr[sid]:].split(b'\0', 1)[0].decode()
                for sid in range(m.nsite)}
    # joint_names는 위에서 이미 생성됨

    def obj_name(objtype, objid):
        """주요 object 타입별 id→name 변환(없으면 raw id 반환)"""
        if objtype == obj_enum.mjOBJ_BODY:
            return body_names.get(objid, str(objid))
        if objtype == obj_enum.mjOBJ_SITE:
            return site_names.get(objid, str(objid))
        if objtype == obj_enum.mjOBJ_JOINT:
            return joint_names.get(objid, str(objid))
        return str(objid)

    # Print sensor table header
    shdr = " id | name                        | type             | dim | adr | target (obj)"
    ssep = "----+-----------------------------+------------------+-----+-----+----------------"
    node.get_logger().info(shdr)
    node.get_logger().info(ssep)

    # Sensor Loop
    for sid in range(m.nsensor):
        adr  = m.name_sensoradr[sid]
        name = m.names[adr:].split(b'\0', 1)[0].decode()

        stype = int(m.sensor_type[sid])
        tstr  = sens2name.get(stype, "Unk")

        dim   = int(m.sensor_dim[sid])
        sadr  = int(m.sensor_adr[sid])

        objtype = int(m.sensor_objtype[sid])
        objid   = int(m.sensor_objid[sid])
        target  = f"{obj2name.get(objtype,'-')}:{obj_name(objtype,objid)}" \
                if objid >= 0 else "-"

        node.get_logger().info(
            f"{sid:3d} | {name:27s} | {tstr:16s} | {dim:3d} | {sadr:3d} | {target}"
        )

      # ---------- Camera table (id, name, mode, resolution) ----------
    node.get_logger().info("")

    chdr = " id | name                        | mode     | resolution"
    csep = "----+-----------------------------+----------+------------"
    node.get_logger().info(chdr)
    node.get_logger().info(csep)

    # mode enum → 문자열 매핑 (가능하면 자동 추출, 없으면 기본 매핑)
    mode_map = {}
    cam_enum = getattr(mujoco, "mjtCamLight", None)
    if cam_enum is not None:
        for attr in dir(cam_enum):
            if attr.startswith("mjCAMLIGHT_"):
                # mjCAMLIGHT_FIXED -> Fixed
                mode_map[getattr(cam_enum, attr)] = attr[10:].title()
    else:
        # fallback (주요 모드)
        mode_map = {0: "Fixed", 1: "Track", 2: "Trackcom"}

    # 해상도: MuJoCo는 per-camera 해상도가 아니라 global offscreen 설정을 사용
    try:
        offw = int(m.vis.global_.offwidth)
        offh = int(m.vis.global_.offheight)
        res_str = f"{offw}x{offh}" if offw > 0 and offh > 0 else "-"
    except Exception:
        res_str = "-"

    ncam = getattr(m, "ncam", 0)
    for cid in range(ncam):
        # name
        cadr  = m.name_camadr[cid]
        cname = m.names[cadr:].split(b'\x00', 1)[0].decode()

        # mode
        if hasattr(m, "cam_mode"):
            try:
                mode_val = int(m.cam_mode[cid])
                mode_str = mode_map.get(mode_val, str(mode_val))
            except Exception:
                mode_str = "-"
        else:
            mode_str = "-"

        node.get_logger().info(f"{cid:3d} | {cname:27s} | {mode_str:8s} | {res_str}")
        
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