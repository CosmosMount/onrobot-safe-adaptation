"""Validation of the canonical Go2 MuJoCo model contract."""

from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from train.core.base import (
    ACTION_SIZE,
    ACTION_SPEC,
    DEFAULT_BASE_HEIGHT,
    DEFAULT_JOINT_POSITION,
    GRAVITY_Z,
    JOINT_NAMES,
    PHYSICS_DT,
)


SDK_MOTOR_ORDER = tuple(JOINT_NAMES)


DEFAULT_GO2_SCENE = (
    Path(__file__).resolve().parents[3]
    / "assets"
    / "robots"
    / "go2"
    / "mjcf"
    / "scene.xml"
)
_VALIDATED_GO2_SCENES: set[Path] = set()


def validate_go2_mjcf_contract(scene_path: str | Path = DEFAULT_GO2_SCENE) -> None:
    """Fail fast when the bridge-indexed Go2 asset violates the SDK contract."""

    scene_path = Path(scene_path).resolve()
    if scene_path in _VALIDATED_GO2_SCENES:
        return
    if not scene_path.is_file():
        raise FileNotFoundError(f"MuJoCo scene does not exist: {scene_path}")
    scene_root = ElementTree.parse(scene_path).getroot()
    include = scene_root.find("include")
    if include is None or not include.get("file"):
        raise ValueError("Canonical MuJoCo scene must include the Go2 MJCF")
    robot_path = (scene_path.parent / include.get("file")).resolve()
    robot_root = ElementTree.parse(robot_path).getroot()

    actuators = robot_root.findall("./actuator/motor")
    actuator_names = tuple(element.get("name") for element in actuators)
    if actuator_names != SDK_MOTOR_ORDER:
        raise ValueError(
            "MuJoCo actuator order must match the SDK contract exactly: "
            f"expected {SDK_MOTOR_ORDER}, got {actuator_names}"
        )
    actuator_joints = tuple(element.get("joint") for element in actuators)
    expected_joints = tuple(f"{name}_joint" for name in SDK_MOTOR_ORDER)
    if actuator_joints != expected_joints:
        raise ValueError(
            "MuJoCo actuators target the wrong joints: "
            f"expected {expected_joints}, got {actuator_joints}"
        )

    sensors = list(robot_root.findall("./sensor/*"))
    expected_sensor_names = tuple(
        f"{name}_{suffix}"
        for suffix in ("pos", "vel", "torque")
        for name in SDK_MOTOR_ORDER
    )
    sensor_names = tuple(element.get("name") for element in sensors[: 3 * ACTION_SIZE])
    if sensor_names != expected_sensor_names:
        raise ValueError(
            "MuJoCo motor sensor order must be position/velocity/torque in SDK "
            f"joint order; got {sensor_names}"
        )
    expected_sensor_tags = (
        ("jointpos",) * ACTION_SIZE
        + ("jointvel",) * ACTION_SIZE
        + ("jointactuatorfrc",) * ACTION_SIZE
    )
    expected_sensor_joints = tuple(
        f"{name}_joint"
        for _ in ("pos", "vel", "torque")
        for name in SDK_MOTOR_ORDER
    )
    for sensor, tag, joint in zip(
        sensors[: 3 * ACTION_SIZE],
        expected_sensor_tags,
        expected_sensor_joints,
    ):
        if sensor.tag != tag or sensor.get("joint") != joint:
            raise ValueError(
                f"MuJoCo sensor {sensor.get('name')} must be <{tag} "
                f"joint='{joint}'>"
            )

    imu_site = robot_root.find(".//site[@name='imu']")
    if imu_site is None:
        raise ValueError("Go2 MJCF is missing the canonical IMU site")
    imu_position = np.fromstring(imu_site.get("pos", "0 0 0"), sep=" ")
    if imu_position.shape != (3,) or not np.allclose(
        imu_position, 0.0, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError(
            "Go2 IMU must sample the shared base_link origin, got "
            f"{imu_position.tolist()}"
        )
    for name in ("imu_quat", "frame_pos", "frame_vel"):
        sensor = robot_root.find(f"./sensor/*[@name='{name}']")
        if (
            sensor is None
            or sensor.get("objtype") != "body"
            or sensor.get("objname") != "base_link"
        ):
            raise ValueError(
                f"MuJoCo sensor {name} must reference the base_link body origin"
            )

    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError(
            "Validating the MuJoCo reset/contact contract requires the project's "
            "mujoco optional dependency"
        ) from exc

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    if not np.isclose(model.opt.timestep, PHYSICS_DT, rtol=0.0, atol=1.0e-12):
        raise ValueError(
            f"MuJoCo timestep must be {PHYSICS_DT}, got {model.opt.timestep}"
        )
    if not np.allclose(
        model.opt.gravity,
        np.asarray([0.0, 0.0, GRAVITY_Z]),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(f"MuJoCo gravity does not match the shared contract")

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    if not np.allclose(
        data.qpos[:7],
        np.asarray([0.0, 0.0, DEFAULT_BASE_HEIGHT, 1.0, 0.0, 0.0, 0.0]),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            "MuJoCo qpos0 must reset base_link to the canonical height and identity pose"
        )
    for index, name in enumerate(SDK_MOTOR_ORDER):
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_joint"
        )
        if joint_id < 0:
            raise ValueError(f"MuJoCo model is missing joint {name}_joint")
        qpos_address = int(model.jnt_qposadr[joint_id])
        expected = float(DEFAULT_JOINT_POSITION[index])
        if not np.isclose(data.qpos[qpos_address], expected, rtol=0.0, atol=1.0e-7):
            raise ValueError(
                f"MuJoCo qpos0 for {name} must be {expected}, "
                f"got {data.qpos[qpos_address]}"
            )

    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id < 0 or not np.allclose(
        model.key_qpos[home_id], data.qpos, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError(
            "The named home keyframe and ordinary mj_resetData qpos0 must be identical"
        )

    compiled_actuator_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
        for index in range(model.nu)
    )
    if compiled_actuator_names != SDK_MOTOR_ORDER:
        raise ValueError(
            "Compiled MuJoCo actuator order differs from the SDK motor order: "
            f"{compiled_actuator_names}"
        )
    expected_ctrlrange = np.tile(
        np.asarray([-ACTION_SPEC.effort_limit, ACTION_SPEC.effort_limit]),
        (ACTION_SIZE, 1),
    )
    if model.nu != ACTION_SIZE or not np.allclose(
        model.actuator_ctrlrange,
        expected_ctrlrange,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            "MuJoCo actuator ctrlrange must be the shared +/-23.5 effort contract"
        )
    if not np.all(np.asarray(model.actuator_ctrllimited, dtype=bool)):
        raise ValueError("Every MuJoCo actuator must enforce its ctrlrange")
    for name in SDK_MOTOR_ORDER:
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_joint"
        )
        dof_address = int(model.jnt_dofadr[joint_id])
        actual_passive = np.asarray(
            [
                model.dof_armature[dof_address],
                model.dof_damping[dof_address],
                model.dof_frictionloss[dof_address],
            ]
        )
        expected_passive = np.asarray(
            [
                ACTION_SPEC.armature,
                ACTION_SPEC.joint_damping,
                ACTION_SPEC.joint_friction,
            ]
        )
        if not np.allclose(
            actual_passive, expected_passive, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError(
                f"MuJoCo passive joint parameters for {name} differ from "
                f"the shared contract: {actual_passive.tolist()}"
            )
    for address, name in enumerate(expected_sensor_names):
        sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sensor_id < 0 or int(model.sensor_adr[sensor_id]) != address:
            raise ValueError(
                f"MuJoCo sensor {name} must occupy sensordata[{address}] for the bridge"
            )

    mujoco.mj_forward(model, data)
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    foot_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, leg)
        for leg in ("FR", "FL", "RR", "RL")
    }
    if floor_id < 0 or -1 in foot_ids:
        raise ValueError("Canonical MuJoCo scene must name the floor and all four feet")
    expected_contact_friction = np.asarray([0.4, 0.02, 0.01])
    for geom_id in {floor_id, *foot_ids}:
        actual_friction = np.asarray(model.geom_friction[geom_id])
        if not np.allclose(
            actual_friction,
            expected_contact_friction,
            rtol=0.0,
            atol=1.0e-12,
        ):
            geom_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
            )
            raise ValueError(
                f"MuJoCo contact friction for {geom_name} must be "
                f"{expected_contact_friction.tolist()}, got "
                f"{actual_friction.tolist()}"
            )
    contacting_feet: set[int] = set()
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        pair = {int(contact.geom1), int(contact.geom2)}
        if floor_id in pair:
            contacting_feet.update(pair & foot_ids)
    if contacting_feet != foot_ids:
        missing = sorted(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, foot_id)
            for foot_id in foot_ids - contacting_feet
        )
        raise ValueError(
            "Ordinary MuJoCo reset must place all four feet in ground contact; "
            f"missing contacts for {missing}"
        )
    _VALIDATED_GO2_SCENES.add(scene_path)

