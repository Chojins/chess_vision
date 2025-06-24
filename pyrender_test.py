import numpy as np
import trimesh
import pyrender
import cv2
import chess_vision
import pickle

def main():
    # Create a simple scene with a pawn mesh
    # Define materials for the mesh
    blue = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.0, 0.0, 1.0, 1.0],  # RGBA (0-1)
        metallicFactor=0.0,
        roughnessFactor=0.5,
    )

    chess_piece = trimesh.load('stl/pawn.stl', force='mesh')
    chess_piece.apply_scale(0.001) # Scale to metres

    mesh = pyrender.Mesh.from_trimesh(chess_piece, material=blue)
    scene = pyrender.Scene()
    scene.add(mesh)

    #camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    
    #get the actual camera pose
    chess_vision.load_saved_transform()

    # Load camera calibration data
    with open('camera_calibration.pkl', 'rb') as f:
        calibration_data = pickle.load(f)
        
    camera_matrix = calibration_data['camera_matrix']

    current_transform = chess_vision.saved_transform.get(chess_vision.WHITE_SIDE_CAMERA)

    pose = (current_transform['rvec'], current_transform['tvec'])
    rvec, tvec = pose
    R, _ = cv2.Rodrigues(rvec)
    T_board = np.eye(4, dtype=np.float32)
    T_board[:3, :3] = R
    T_board[:3, 3] = tvec.squeeze()

    T_camera_in_board = np.linalg.inv(T_board)
    print("Camera pose in board axes:\n", T_camera_in_board)

    #Convert CV camera axes  ➜  OpenGL camera axes
    # Create rotation matrix for 180° around X
    R_x180 = np.array([[1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]], dtype=np.float32)
    
    # Create shift transform
    T_xrotate = np.eye(4, dtype=np.float32)
    T_xrotate[:3, :3] = R_x180
    print("rotate X 180 degrees:\n", T_xrotate)
    
    #convert the camera pose to OpenGL axes convention
    T_render_camera = T_camera_in_board @ T_xrotate        
    print("Camera pose in OpenGL axes:\n", T_render_camera)

    # --- Add to the scene -----------------------------------------------------
    fx, fy, cx, cy = camera_matrix[0,0], camera_matrix[1,1], camera_matrix[0,2], camera_matrix[1,2]   # intrinsics from cv2
    camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy, znear=0.01, zfar=10.0)
    scene.add(camera, pose=T_render_camera)

    #directional light off-axis
    light_pose = np.eye(4)
    light_pose[:3, 3] = [-40, 60, 80]            # above & to the left
    sun = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
    scene.add(sun, pose=light_pose)

    r = pyrender.OffscreenRenderer(640, 480)
    color, depth = r.render(scene)

    # OpenCV expects BGR, pyrender gives RGB
    cv2.imshow("Color", cv2.cvtColor(color, cv2.COLOR_RGB2BGR))

    # Normalize depth for display
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_display = depth_normalized.astype(np.uint8)
    cv2.imshow("Depth", depth_display)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()