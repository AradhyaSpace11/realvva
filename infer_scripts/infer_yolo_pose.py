import time
import pybullet as p
import pybullet_data
import numpy as np
import cv2
import torch
import os
from ultralytics import YOLO

# --- CONFIG ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
MODEL_PATH = os.path.join(ROOT_DIR, "trained_models/yolo_pose_model.pt")
RENDER_W, RENDER_H = 640, 640

def get_robot_view():
    view = p.computeViewMatrixFromYawPitchRoll([0.2, 0, 0.1], 1.2, 45, -40, 0, 2)
    proj = p.computeProjectionMatrixFOV(50, 1.0, 0.1, 4.0)
    _, _, rgb, _, _ = p.getCameraImage(RENDER_W, RENDER_H, view, proj, shadow=1, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    frame = np.reshape(rgb, (RENDER_H, RENDER_W, 4))[:, :, :3]
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

def main():
    # 1. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        print("Please run training_scripts/train_yolo.py first!")
        return
        
    print(f"Loading YOLO Pose Model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    # 2. Setup Sim
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    plane_id = p.loadURDF("plane.urdf")
    p.changeVisualShape(plane_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])
    
    robot = p.loadURDF("../3D_models/gripper_arm.urdf", basePosition=[0, 0, 0], useFixedBase=True)
    
    # Randomize Cube
    rand_x = 0.4 + np.random.uniform(-0.1, 0.1)
    rand_y = 0.0 + np.random.uniform(-0.1, 0.1)
    cube_id = p.loadURDF("cube.urdf", basePosition=[rand_x, rand_y, 0.05], globalScaling=0.05)
    p.changeVisualShape(cube_id, -1, rgbaColor=[1, 0, 0, 1])
    
    # Joints
    joints = [0, 1, 2, 3, 4, 5]
    for i in joints:
        p.resetJointState(robot, i, 0)
        
    print("Starting YOLO Inference (Visualization Only)...")
    print("Move the robot using the GUI sliders (if verified) or just watch it verify detections.")
    
    # Optional: Enable Debug Sliders to move robot and check detection robustness
    sliders = []
    for i in joints:
        sliders.append(p.addUserDebugParameter(f"Joint {i}", -3.14, 3.14, 0))
    
    while True:
        # User Interactivity
        target_pos = []
        for s in sliders:
            target_pos.append(p.readUserDebugParameter(s))
        p.setJointMotorControlArray(robot, joints, p.POSITION_CONTROL, targetPositions=target_pos)
        p.stepSimulation()
        
        # 1. Get View
        img = get_robot_view()
        
        # 2. Inference
        results = model(img, verbose=False)
        
        # 3. Visualize
        # Ultralytics has built-in plot() but we can draw manually for clarity
        annotated_frame = results[0].plot()
        
        # Extract Keypoints if needed for downstream
        if results[0].keypoints is not None and results[0].keypoints.xy.numel() > 0:
             # kpts = results[0].keypoints.xy[0].cpu().numpy() # [6, 2]
             pass

        cv2.imshow("YOLO Pose Inference", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    p.disconnect()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
