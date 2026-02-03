import time
import pybullet as p
import pybullet_data
import numpy as np
import cv2
import math
import threading
import serial

# --- 1. SIMULATION SETUP ---
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

plane_id = p.loadURDF("plane.urdf")
p.changeVisualShape(plane_id, -1, textureUniqueId=-1)
p.changeVisualShape(plane_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])

robot = p.loadURDF("../3D_models/gripper_arm.urdf", basePosition=[0, 0, 0], useFixedBase=True)
cube_id = p.loadURDF("cube.urdf", basePosition=[0.4, 0, 0.05], globalScaling=0.05)
p.changeVisualShape(cube_id, -1, rgbaColor=[0, 0.5, 1, 1])

p.changeDynamics(robot, 4, lateralFriction=2.0)
p.changeDynamics(robot, 5, lateralFriction=2.0)

# --- 2. SERIAL INPUT THREAD (ESP32) ---
# ESP32 sends: yaw,shoulder,elbow,endlink,button
serial_state = {"yaw": 0.0, "shoulder": 0.0, "elbow": 0.0, "end": 0.0, "button": 0}

def serial_listener(port="/dev/ttyUSB0", baud=115200):
  while True:
    try:
      ser = serial.Serial(port, baud, timeout=1)
      time.sleep(1.0)
      while True:
        line = ser.readline().decode(errors="ignore").strip()
        if not line:
          continue
        try:
          y, s, e, en, b = line.split(",")
          serial_state["yaw"] = float(y)
          serial_state["shoulder"] = float(s)
          serial_state["elbow"] = float(e)
          serial_state["end"] = float(en)
          serial_state["button"] = int(b)
        except Exception:
          pass
    except Exception:
      time.sleep(1.0)

threading.Thread(target=serial_listener, daemon=True).start()

# --- 3. CONTROL & CAMERA SETUP ---
joints = [0, 1, 2, 3, 4, 5]
pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
LIMITS = [(-3.14, 3.14), (-3.14, 3.14), (-3.14, 3.14), (-3.14, 3.14), (0, 0.5), (-0.5, 0)]

RENDER_W, RENDER_H = 400, 400
SKIP_FRAMES = 5
frame_counter = 0

def get_frame(eye, target, up):
  view = p.computeViewMatrix(eye, target, up)
  proj = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 5.0)
  _, _, rgb, _, _ = p.getCameraImage(RENDER_W, RENDER_H, view, proj, shadow=0, renderer=p.ER_BULLET_HARDWARE_OPENGL)
  return cv2.cvtColor(np.reshape(rgb, (RENDER_H, RENDER_W, 4))[:, :, :3], cv2.COLOR_RGB2BGR)

def deg_to_rad(deg):
  return np.clip(math.radians(deg), -math.pi, math.pi)

# --- 4. MAIN LOOP ---
print("System Active! Pots control arm. Button controls gripper (open by default).")

while True:
  # Arm from pots (signed degrees -> radians)
  pos[0] = deg_to_rad(serial_state["yaw"])
  pos[1] = deg_to_rad(serial_state["shoulder"])
  pos[2] = -deg_to_rad(serial_state["elbow"])
  pos[3] = deg_to_rad(serial_state["end"])

  # Gripper: open always, close only when button pressed
  if serial_state["button"] == 1:
    pos[4] = 0.5
    pos[5] = -0.5
  else:
    pos[4] = 0.0
    pos[5] = 0.0

  # Clamp
  for i in range(6):
    pos[i] = np.clip(pos[i], LIMITS[i][0], LIMITS[i][1])

  # Physics
  p.setJointMotorControlArray(robot, joints, p.POSITION_CONTROL, targetPositions=pos, forces=[100, 100, 100, 100, 40, 40])
  p.stepSimulation()
  frame_counter += 1

  # Camera
  if frame_counter % SKIP_FRAMES == 0:
    link_state = p.getLinkState(robot, 2)
    h = link_state[0][2]
    ang = pos[0] + (math.pi / 2)

    side_img = get_frame([1.2 * math.cos(ang), 1.2 * math.sin(ang), h], [0, 0, h], [0, 0, 1])
    top_img = get_frame([0.1, 0, 1.2], [0.1, 0, 0], [0, 1, 0])

    combined_view = np.vstack((side_img, top_img))
    cv2.imshow("Robot Monitoring", combined_view)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

  time.sleep(1 / 240)

cv2.destroyAllWindows()
p.disconnect()