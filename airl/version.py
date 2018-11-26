import mujoco_py

MUJOCO_VERSION = '1.3.1'

if hasattr(mujoco_py, 'version'):
    MUJOCO_VERSION = mujoco_py.version.get_version()

VERSION = 0.1