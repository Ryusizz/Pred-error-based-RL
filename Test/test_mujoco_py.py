import mujoco_py
import os
mj_path, _ = mujoco_py.utils.discover_mujoco()
print("mujoco_path", mj_path)
# xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')
xml_path = "/home/reasonable/mujoco_ur5_model/example.xml"
model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)

print(sim.data.qpos)
# [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

sim.step()
print(sim.data.qpos)