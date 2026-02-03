import time
import pybullet as p
import pybullet_data
import numpy as np
import cv2
import math

# --- CONFIG ---
RENDER_W, RENDER_H = 640, 640
# Camera Config (Fixed)
CAM_TARGET = [0.4, 0.0, 0.0]
CAM_DIST = 1.2
CAM_YAW = 45
CAM_PITCH = -40
CAM_ROLL = 0
FOV = 50

# World Limits (approximate, based on scene)
X_MIN, X_MAX = 0.2, 0.6
Y_MIN, Y_MAX = -0.2, 0.2

# --- UTILS ---
def get_robot_view():
    view = p.computeViewMatrixFromYawPitchRoll(CAM_TARGET, CAM_DIST, CAM_YAW, CAM_PITCH, CAM_ROLL, 2)
    proj = p.computeProjectionMatrixFOV(FOV, 1.0, 0.1, 4.0)
    _, _, rgb, _, _ = p.getCameraImage(RENDER_W, RENDER_H, view, proj, shadow=1, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    frame = np.reshape(rgb, (RENDER_H, RENDER_W, 4))[:, :, :3]
    return frame.astype(np.uint8)

def find_red_cube(img):
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # Red has two ranges in HSV (0-10 and 170-180)
    lower1 = np.array([0, 100, 100])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 100, 100])
    upper2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = mask1 + mask2
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get largest contour
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy), mask
    return None, mask

def map_pixel_to_world(u, v):
    # Heuristic Calibration
    # We know the Camera is looking at (0.4, 0, 0) -> Center of Image (320, 320)
    # We need to determine the scale.
    # At dist=1.2, FOV=50, the visible width at Z=0 is approx:
    # Width ~ 2 * dist * tan(FOV/2) ~ 2 * 1.2 * 0.46 ~ 1.1 meters?
    # Actually, pitch=-40 compresses Y.
    
    # Let's use a simple linear map determined empirically or logically.
    # Image Center (320, 320) -> World (0.4, 0.0)
    # Image X axis corresponds roughly to World Y axis (rotated 90deg due to cam yaw=45?)
    
    # Let's interactively calibrate or use a "Jacobian" style approach.
    # Approach: 
    #   Pixel Error (du, dv) -> World Error (dx, dy)
    #   We will use a PROPORTIONAL CONTROLLER instead of absolute mapping.
    #   Target (u, v) is the cube.
    #   Current End Effector (u_ee, v_ee) can be found by projecting world pos to screen!
    pass

def project_world_to_pixel(pos):
    # pos: [x, y, z]
    view = p.computeViewMatrixFromYawPitchRoll(CAM_TARGET, CAM_DIST, CAM_YAW, CAM_PITCH, CAM_ROLL, 2)
    proj = p.computeProjectionMatrixFOV(FOV, 1.0, 0.1, 4.0)
    viewMat = np.array(view).reshape(4,4).T
    projMat = np.array(proj).reshape(4,4).T
    
    pos_h = np.array([pos[0], pos[1], pos[2], 1.0])
    clip_pos = projMat @ viewMat @ pos_h
    clip_pos /= clip_pos[3] # perspective divide
    
    # Clip (-1, 1) -> Pixel (0, W)
    u = (clip_pos[0] + 1) / 2 * RENDER_W
    v = (1 - clip_pos[1]) / 2 * RENDER_H # Y is inverted in screens
    
    return int(u), int(v)

# --- MAIN ---
def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    plane_id = p.loadURDF("plane.urdf")
    p.changeVisualShape(plane_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])
    
    robot = p.loadURDF("../3D_models/gripper_arm.urdf", basePosition=[0, 0, 0], useFixedBase=True)
    end_effector_index = 5 # Approx tip

    # Change Robot Color to Grey to avoid Red Detection Confusion
    for k in range(p.getNumJoints(robot)):
        p.changeVisualShape(robot, k, rgbaColor=[0.5, 0.5, 0.5, 1])

    
    # Randomize Cube
    rand_x = 0.4 + np.random.uniform(-0.1, 0.1) # Wider range
    rand_y = 0.0 + np.random.uniform(-0.15, 0.15)
    cube_id = p.loadURDF("cube.urdf", basePosition=[rand_x, rand_y, 0.05], globalScaling=0.05)
    p.changeVisualShape(cube_id, -1, rgbaColor=[1, 0, 0, 1])
    print(f"Target at: {rand_x:.2f}, {rand_y:.2f}")
    
    # Home
    joints = [0, 1, 2, 3, 4, 5]
    for i in joints:
        p.resetJointState(robot, i, 0)
        
    print("Starting Visual Servoing (Target Chasing)...")
    
    # Control Parameters
    Kp = 0.001 # Pixel error to meters gain (Start small!)
    
    # Current Desired Position (Starts at Home)
    current_target_world = [0.2, 0.2, 0.15] # Start somewhere high
    
    while True:
        # 1. Vision
        img = get_robot_view()
        cube_center, mask = find_red_cube(img)
        
        if cube_center:
            # 2. Get Current EE Position in Image
            ee_state = p.getLinkState(robot, end_effector_index)
            ee_pos_world = ee_state[0]
            ee_center = project_world_to_pixel(ee_pos_world)
            
            # 3. Calculate Error in IMAGE SPACE
            du = cube_center[0] - ee_center[0]
            dv = cube_center[1] - ee_center[1]
            
            # 4. Map Image Error to World Update (Jacobian Approx)
            # We need to know orientation.
            # Camera Yaw=45.
            #   Screen X (u) ~ -World Y (roughly) + World X (roughly)
            #   Screen Y (v) ~ -World Z (roughly) + World X?
            
            # Let's cheat slightly and use IK purely on World Coords? 
            # NO, the user wants "Visual" servoing.
            # Let's align the axes heuristically.
            # Rotation Matrix of Camera (roughly):
            # The camera looks at (0.4, 0, 0) from (0.2, 0, 0.1) -> Vector (0.2, 0, -0.1).
            # Wait, PyBullet ComputeViewMatrixFromYawPitchRoll handles the complex rotation.
            
            # ALTERNATIVE: Use coordinate descent ("Wiggle Test")? No, too slow.
            
            # Let's assume standard camera alignment and derive it.
            # Yaw=45 -> Rotated 45 deg around Z.
            # Image Right (+u) -> World Right-ish relative to camera.
            # Image Down (+v) -> World Down (-Z) or Forward (+X)?
            # Pitch = -40 (Looking Down). So +v is definitely "Closer/Down".
            
            # Heuristic Update:
            # Let's try to update World X/Y based on u/v.
            # To go Right on Screen (+u): Move Robot Y- / X+ ?
            # To go Down on Screen (+v): Move Robot Z- / X+ ?
            
            # SIMPLER METHOD: "Chase the Cube 3D"
            # If we know Z_table = 0.05.
            # We can project the Single Pixel (u, v) + Z_world -> X_world, Y_world.
            # Ray Casting!
            
            # 5. Ray Cast Method (Robust)
            view = p.computeViewMatrixFromYawPitchRoll(CAM_TARGET, CAM_DIST, CAM_YAW, CAM_PITCH, CAM_ROLL, 2)
            proj = p.computeProjectionMatrixFOV(FOV, 1.0, 0.1, 4.0)
            viewMat = np.array(view).reshape(4,4).T
            projMat = np.array(proj).reshape(4,4).T
            
            # NDC
            x_ndc = (2.0 * cube_center[0] / RENDER_W) - 1.0
            y_ndc = 1.0 - (2.0 * cube_center[1] / RENDER_H)
            z_ndc = 1.0 # Far plane? Doesn't matter for ray dir
            
            # Unproject
            inv_proj = np.linalg.inv(projMat)
            inv_view = np.linalg.inv(viewMat)
            
            clip_coords = np.array([x_ndc, y_ndc, -1.0, 1.0]) # Near plane
            eye_coords = inv_proj @ clip_coords
            eye_coords = np.array([eye_coords[0], eye_coords[1], -1.0, 0.0]) # Direction?
            
            # Wait, simpler standard unproject:
            # Ray Start = Camera Pos
            # Ray End = Unproject(x_ndc, y_ndc, 1.0)
            
            # Camera Position from params:
            # FromYawPitchRoll implies the camera is on a sphere of radius DIST around TARGET.
            # It's better to just read the matrix translation.
            cam_pos = inv_view[:3, 3]
            
            # Ray Dir
            clip_target = np.array([x_ndc, y_ndc, 1.0, 1.0])
            eye_target = inv_proj @ clip_target
            eye_target /= eye_target[3]
            world_target = inv_view @ eye_target
            
            ray_dir = world_target[:3] - cam_pos
            ray_dir /= np.linalg.norm(ray_dir)
            
            # Intersect with Plane Z = 0.025 (Cube center height)
            # P = O + t * D
            # P.z = 0.025 => O.z + t * D.z = 0.025 => t = (0.025 - O.z) / D.z
            if ray_dir[2] != 0:
                t = (0.025 - cam_pos[2]) / ray_dir[2]
                world_x = cam_pos[0] + t * ray_dir[0]
                world_y = cam_pos[1] + t * ray_dir[1]
                
                # Move towards this spot
                # Smooth approach
                target = [world_x, world_y, 0.05] # Hover
                
                # IK
                j_vals = p.calculateInverseKinematics(robot, end_effector_index, target)
                
                p.setJointMotorControlArray(robot, joints, p.POSITION_CONTROL, targetPositions=j_vals, forces=[100]*6)
                
                # Viz
                p.addUserDebugLine([world_x, world_y, 0], [world_x, world_y, 0.2], [0, 1, 0], 1, 0.1)
                
            cv2.circle(img, cube_center, 5, (0, 255, 0), 2)
            cv2.circle(img, ee_center, 3, (255, 0, 0), 2)
            
        cv2.imshow("Robot View (Red Detection)", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        # cv2.imshow("Mask", mask)
        
        p.stepSimulation()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(1/240.)

    p.disconnect()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
