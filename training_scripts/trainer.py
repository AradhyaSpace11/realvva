import time
import pybullet as p
import pybullet_data
import numpy as np
import cv2
import math
import threading
import serial
import os

# --- 1. SETUP ---
for folder in ["data/demovideos", "data/jointdata"]:
    os.makedirs(folder, exist_ok=True)

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

plane_id = p.loadURDF("plane.urdf")
p.changeVisualShape(plane_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1], textureUniqueId=-1)

robot = p.loadURDF("../3D_models/gripper_arm.urdf", basePosition=[0, 0, 0], useFixedBase=True)
cube_id = p.loadURDF("cube.urdf", basePosition=[0.4, 0, 0.05], globalScaling=0.05)
p.changeVisualShape(cube_id, -1, rgbaColor=[1, 0, 0, 1])

# --- 2. FRICTION ENHANCEMENT ---
# Increase friction on both the gripper fingers and the object
# Links 4 and 5 are usually the gripper fingers in common URDFs
for link_idx in [4, 5]:
    p.changeDynamics(robot, link_idx, 
                     lateralFriction=5.0, 
                     spinningFriction=1.0, 
                     rollingFriction=1.0,
                     contactStiffness=10000,
                     contactDamping=1)

p.changeDynamics(cube_id, -1, 
                 lateralFriction=5.0, 
                 spinningFriction=1.0, 
                 rollingFriction=1.0)

# --- 3. HIGH-RES CAMERA CONFIG ---
RENDER_W, RENDER_H = 640, 640 # Increased resolution for better VVA training
FPS = 30

def get_diagonal_view():
    view = p.computeViewMatrixFromYawPitchRoll([0.2, 0, 0.1], 1.2, 45, -40, 0, 2)
    # Adjust FOV and clipping for better clarity
    proj = p.computeProjectionMatrixFOV(50, 1.0, 0.1, 4.0)
    
    # Use shadows and high-quality renderer for the recording
    _, _, rgb, _, _ = p.getCameraImage(RENDER_W, RENDER_H, view, proj, 
                                       shadow=1, 
                                       renderer=p.ER_BULLET_HARDWARE_OPENGL)
    
    frame = np.reshape(rgb, (RENDER_H, RENDER_W, 4))[:, :, :3]
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

# --- 4. SERIAL THREAD (Existing Logic) ---
serial_state = {"yaw": 0.0, "shoulder": 0.0, "elbow": 0.0, "end": 0.0, "button": 0}
def serial_listener(port="/dev/ttyUSB0", baud=115200):
    while True:
        try:
            ser = serial.Serial(port, baud, timeout=1)
            while True:
                line = ser.readline().decode(errors="ignore").strip()
                if not line: continue
                try:
                    y, s, e, en, b = line.split(",")
                    serial_state.update({"yaw": float(y), "shoulder": float(s), "elbow": float(e), "end": float(en), "button": int(b)})
                except: pass
        except: time.sleep(1.0)

threading.Thread(target=serial_listener, daemon=True).start()

# --- 5. RECORDING LOGIC ---
input("System ready. Press ENTER to start 5s countdown...")
for i in range(5, 0, -1):
    print(f"Recording starts in {i}...")
    time.sleep(1)
 
session_name = f"rec_{int(time.time())}"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter(f"data/demovideos/{session_name}.mp4", fourcc, FPS, (RENDER_W, RENDER_H))

print("RECORDING...")
joints = [0, 1, 2, 3, 4, 5]
LIMITS = [(-3.14, 3.14), (-3.14, 3.14), (-3.14, 3.14), (-3.14, 3.14), (0, 0.5), (-0.5, 0)]
prev_pos = [0.0] * 6

try:
    with open(f"data/jointdata/{session_name}_joints.csv", "w") as f:
        f.write("timestamp,j0,j1,j2,j3,j4,j5,d0,d1,d2,d3,d4,d5\n")
        start_t = time.time()
        
        while True:
            current_pos = [
                np.clip(math.radians(serial_state["yaw"]), *LIMITS[0]),
                np.clip(math.radians(serial_state["shoulder"]), *LIMITS[1]),
                np.clip(-math.radians(serial_state["elbow"]), *LIMITS[2]),
                np.clip(math.radians(serial_state["end"]), *LIMITS[3]),
                0.5 if serial_state["button"] == 1 else 0.0,
                -0.5 if serial_state["button"] == 1 else 0.0
            ]
            
            deltas = [c - p for c, p in zip(current_pos, prev_pos)]
            prev_pos = current_pos[:]

            p.setJointMotorControlArray(robot, joints, p.POSITION_CONTROL, targetPositions=current_pos, forces=[150]*4 + [60]*2)
            p.stepSimulation()

            img = get_diagonal_view()
            out_video.write(img)
            
            f.write(f"{time.time()-start_t}," + ",".join(map(str, current_pos + deltas)) + "\n")
            
            cv2.imshow("High-Res Stream", img)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            time.sleep(1/FPS)

finally:
    out_video.release()
    cv2.destroyAllWindows()
    p.disconnect()