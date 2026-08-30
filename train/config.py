from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MUJOCO_SCENE = PROJECT_ROOT / "assets" / "robots" / "go2" / "mjcf" / "scene.xml"
