import argparse
import cv2
import numpy as np
import glob
import pickle
import json
import datetime
import chess
from pathlib import Path
from board_3d_overlay import (
    load_piece_models,
    render_board_state,
    generate_board_overlay,
    composite_overlay,
)
# Maintain previous public name
render_board_overlay = render_board_state

SQUARE_SIZE = 22.5  # millimeters
square_size_m = SQUARE_SIZE / 1000.0  # Convert mm to meters
CHESSBOARD_SIZE = (7, 7)

# Resolve resource paths relative to this script so it works from any CWD
BASE_DIR = Path(__file__).resolve().parent
CALIBRATION_PATH = BASE_DIR / "camera_calibration.pkl"
BOARD_TRANSFORM_PATH = BASE_DIR / "board_transform.json"

# Load camera calibration data
with open(CALIBRATION_PATH, 'rb') as f:
    calibration_data = pickle.load(f)
    
camera_matrix = calibration_data['camera_matrix']
dist_coeffs = calibration_data['dist_coeffs']

REAL_WHITE_CAMERA = 2
REAL_BLACK_CAMERA = 0

SIM_WHITE_CAMERA = "/World/so100_blue/gripper/Camera"
SIM_BLACK_CAMERA = "/World/so100_red/gripper/Camera"

REAL_MODE = "real"
SIM_MODE = "sim"

saved_transform = None

def _normalize_transform_data(transform_data):
    """Ensure the transform data always contains mode-specific dictionaries."""
    if not isinstance(transform_data, dict):
        return {}

    # Detect legacy format (no explicit mode separation)
    legacy_keys = [
        key for key in list(transform_data.keys())
        if key not in (REAL_MODE, SIM_MODE)
    ]

    if legacy_keys:
        legacy_data = {key: transform_data[key] for key in legacy_keys}
        for key in legacy_keys:
            transform_data.pop(key, None)
        transform_data.setdefault(REAL_MODE, {}).update(legacy_data)

    transform_data.setdefault(REAL_MODE, {})
    transform_data.setdefault(SIM_MODE, {})
    return transform_data

def load_saved_transform(mode, camera_lookup):
    """
    Load and store the transform data for both cameras
    """
    global saved_transform
    camera_keys = list(camera_lookup.keys())
    saved_transform = {key: None for key in camera_keys}
    try:
        with open(BOARD_TRANSFORM_PATH, 'r') as f:
            raw_data = json.load(f)

        data = _normalize_transform_data(raw_data)
        mode_data = data.get(mode, {}) if isinstance(data, dict) else {}

        for key in camera_keys:
            key_str = str(key)
            if key_str in mode_data:
                entry = mode_data[key_str]
            elif key in mode_data:
                entry = mode_data[key]
            else:
                entry = None

            if entry is None:
                saved_transform[key] = None
                continue

            saved_transform[key] = {
                'inner_corners': np.array(entry['inner_corners'], dtype=np.float32),
                'board_size': entry['board_size'],
                'square_size': entry['square_size'],
                'rvec': np.array(entry['rvec']),
                'tvec': np.array(entry['tvec']),
                'use_white_side': entry.get(
                    'use_white_side',
                    camera_lookup[key].get('use_white_side', False)
                )
            }
        print("Loaded saved transforms from file")
    except Exception as e:
        print(f"Error loading transform data: {e}")
        saved_transform = {key: None for key in camera_keys}


class OpenCVCameraSource:
    """Wrapper around ``cv2.VideoCapture`` that provides a consistent interface."""

    def __init__(self, camera_index):
        self.camera_index = camera_index
        self.cap = None

    def open(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.camera_index)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return self.cap.isOpened()

    def ensure_ready(self):
        if self.cap is None or not self.cap.isOpened():
            return self.open()
        return True

    def read(self):
        if not self.ensure_ready():
            return False, None
        return self.cap.read()

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def is_opened(self):
        return self.cap is not None and self.cap.isOpened()


class IsaacSimCameraSource:
    """Camera interface for Isaac Sim RGB sensors."""

    def __init__(self, prim_path):
        self.prim_path = prim_path
        try:
            from omni.isaac.sensor import Camera  # pylint: disable=import-error
        except ImportError as exc:
            raise RuntimeError(
                "Isaac Sim camera support requires running inside an Isaac Sim Python environment."
            ) from exc

        self._camera = Camera(prim_path=prim_path)
        self._camera.initialize()

    def ensure_ready(self):
        return True

    def is_opened(self):
        return True

    def read(self):
        rgba = self._camera.get_rgba()
        if rgba is None:
            return False, None

        frame = np.array(rgba)
        if frame.dtype != np.uint8:
            frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)

        if frame.shape[-1] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        return True, frame

    def release(self):
        # Isaac Sim cameras do not require explicit release operations.
        return

def find_chessboard_corners(img, use_white_side=True):
    """
    Detect chessboard corners using OpenCV's built-in functions.
    """
    square_size_m = SQUARE_SIZE / 1000.0  # Convert mm to meters
    
    # Try finding corners on original image first
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_inv = cv2.bitwise_not(gray)
    
    ret, corners = cv2.findChessboardCorners(
        gray_inv, 
        (7, 7),
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
              cv2.CALIB_CB_NORMALIZE_IMAGE +
              cv2.CALIB_CB_EXHAUSTIVE
    )
    
    if not ret:
        # If failed, try with undistorted image
        undistorted = cv2.undistort(img, camera_matrix, dist_coeffs)
        gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
        gray_inv = cv2.bitwise_not(gray)
        
        ret, corners = cv2.findChessboardCorners(
            gray_inv, 
            (7, 7),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                  cv2.CALIB_CB_NORMALIZE_IMAGE +
                  cv2.CALIB_CB_EXHAUSTIVE
        )
    
    if not ret:
        raise ValueError("Could not detect chessboard corners")
    
    # Refine corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    
    # Create 3D points for the chessboard pattern
    pattern_size = (7, 7)
    objp = np.zeros((np.prod(pattern_size), 3), np.float32)
    objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    objp *= square_size_m
    
    # Find black corner squares and determine orientation
    corner_pts = corners.reshape(7, 7, 2)
    
    def get_square_color(img, corners):
        # Convert corners to integer coordinates
        corners = corners.astype(np.int32)
        
        # Calculate center point of the square
        center_x = int((corners[0][0] + corners[1][0] + corners[2][0] + corners[3][0]) // 4)
        center_y = int((corners[0][1] + corners[1][1] + corners[2][1] + corners[3][1]) // 4)
        
        # Ensure coordinates are within image bounds
        height, width = img.shape[:2]
        center_x = max(2, min(center_x, width-3))
        center_y = max(2, min(center_y, height-3))
        
        # Sample a small region around the center (5x5 pixels)
        roi = img[center_y-2:center_y+3, center_x-2:center_x+3]
        return np.mean(roi)

    # Convert image to grayscale if it isn't already
    if len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img
    
    # Check colors of all corner squares
    corner_squares = [
        # [corners for square], y-coordinate, x-coordinate
        (np.array([
            corner_pts[-1, 0],     # bottom-left corner
            corner_pts[-1, 1],     # bottom-right corner
            corner_pts[-2, 1],     # top-right corner
            corner_pts[-2, 0]      # top-left corner
        ]), corner_pts[-1, 0][1], corner_pts[-1, 0][0]),  # y and x coords
        
        (np.array([
            corner_pts[-1, -2],    # rotated 90° clockwise
            corner_pts[-1, -1],
            corner_pts[-2, -1],
            corner_pts[-2, -2]
        ]), corner_pts[-1, -1][1], corner_pts[-1, -1][0]),
        
        (np.array([
            corner_pts[1, -1],     # rotated 180°
            corner_pts[1, -2],
            corner_pts[0, -2],
            corner_pts[0, -1]
        ]), corner_pts[0, -1][1], corner_pts[0, -1][0]),
        
        (np.array([
            corner_pts[1, 0],      # rotated 270°
            corner_pts[1, 1],
            corner_pts[0, 1],
            corner_pts[0, 0]
        ]), corner_pts[0, 0][1], corner_pts[0, 0][0])
    ]
    
    # Find black corner squares
    black_corners = []
    for i, (square_corners, y_coord, x_coord) in enumerate(corner_squares):
        color = get_square_color(gray_img, square_corners)
        if color < 128:  # if square is black
            black_corners.append((i, y_coord, x_coord))
    
    if not black_corners:
        raise ValueError("No black corners found!")
    
    # Choose the black corner based on parameter
    if use_white_side:
        # Use bottom-left black corner (highest y, lowest x)
        rotations_needed = black_corners[np.argmax([y - x/1000 for _, y, x in black_corners])][0]
    else:
        # Use top-right black corner (lowest y, highest x)
        rotations_needed = black_corners[np.argmin([y + x/1000 for _, y, x in black_corners])][0]
    
    # After finding rotations_needed, rotate the object points accordingly
    objp = objp.reshape(7, 7, 3)
    objp = np.rot90(objp, k=-rotations_needed)
    objp = objp.reshape(-1, 3)
    
    # Flip y-coordinates to match OpenCV's coordinate system
    objp[:, 1] = 6 * square_size_m - objp[:, 1]
    
    # Get initial pose
    ret, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)
    
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    
    # Get the Z axis direction (third column of rotation matrix)
    z_axis = R[:, 2]
    
    # If Z axis is pointing down, flip the pose
    if z_axis[2] < 0:
        # Rotate 180 degrees around Y axis to flip Z direction
        R_flip = np.array([[-1, 0, 0],
                          [0, 1, 0],
                          [0, 0, -1]], dtype=np.float32)
        R = R @ R_flip
        
        # Convert back to rotation vector
        rvec, _ = cv2.Rodrigues(R)
    
    # Get final pose with corrected orientation
    ret, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs, 
                                  rvec=rvec, tvec=tvec, useExtrinsicGuess=True)
      
    # Calculate distance to board origin
    distance_to_board = np.linalg.norm(tvec)  # Distance in meters
    distance_mm = distance_to_board * 1000  # Convert to millimeters
    #print(f"Distance to board: {distance_mm:.1f}mm")
    
    # If we've found the incorrect corner, we need to transform to the correct corner
    DISTANCE_THRESHOLD = 250  # mm
    if (use_white_side and distance_mm > DISTANCE_THRESHOLD) or \
       (not use_white_side and distance_mm <= DISTANCE_THRESHOLD):
        # Create shift transform (6 squares in x and y)
        square_size_m = SQUARE_SIZE / 1000.0  # Convert mm to meters
        tvec_shift = np.array([[6 * square_size_m], 
                             [6 * square_size_m], 
                             [0]], dtype=np.float32)
        
        # Create rotation matrix for 180° around Z
        R_z180 = np.array([[-1, 0, 0],
                          [0, -1, 0],
                          [0, 0, 1]], dtype=np.float32)
        
        # Create homogeneous transformation matrices
        # First create original transform
        R_orig, _ = cv2.Rodrigues(rvec)
        T_orig = np.eye(4, dtype=np.float32)
        T_orig[:3, :3] = R_orig
        T_orig[:3, 3:4] = tvec
        
        # Create shift transform
        T_shift = np.eye(4, dtype=np.float32)
        T_shift[:3, :3] = R_z180
        T_shift[:3, 3:4] = tvec_shift
        
        # Combine transforms
        T_final = T_orig @ T_shift
        
        # Extract new rotation and translation
        tvec = T_final[:3, 3:4]
        R = T_final[:3, :3]
        rvec, _ = cv2.Rodrigues(R)

    # Shift the origin to the outer edge by moving 1 square size in -x and -y
    tvec_shift = np.array([[-square_size_m], 
                           [-square_size_m], 
                           [0]], dtype=np.float32)
    
    # Create homogeneous transformation matrices
    # First create original transform
    R_orig, _ = cv2.Rodrigues(rvec)
    T_orig = np.eye(4, dtype=np.float32)
    T_orig[:3, :3] = R_orig
    T_orig[:3, 3:4] = tvec
    
    # Create shift transform
    T_shift = np.eye(4, dtype=np.float32)
    T_shift[:3, 3:4] = tvec_shift

    # Combine transforms
    T_final = T_orig @ T_shift
    
    # Extract new rotation and translation
    tvec = T_final[:3, 3:4]
    R = T_final[:3, :3]
    rvec, _ = cv2.Rodrigues(R)
    
    # Continue with the rest of the corner processing for the board transform
    corner_pts = np.rot90(corner_pts, k=-rotations_needed)
    
    # Calculate sizes for proper 8x8 board
    inner_squares = 6  # Number of squares we actually see (6x6)
    board_inner_size = 600  # Size for the 6x6 inner board
    square_size = board_inner_size // inner_squares  # Size of each square
    board_full_size = square_size * 8  # 8 squares total
    
    # Get corners of detected inner board
    inner_corners = np.float32([
        corner_pts[0, 0],      # top-left
        corner_pts[0, -1],     # top-right
        corner_pts[-1, -1],    # bottom-right
        corner_pts[-1, 0]      # bottom-left
    ])
    
    # Draw debug visualization of inner corners
    frame_draw = img.copy()
    for corner in inner_corners:
        cv2.circle(frame_draw, tuple(corner.astype(int)), 5, (0, 0, 255), -1)
    pts = inner_corners.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(frame_draw, [pts], True, (0, 255, 0), 2)
    
    return inner_corners, board_full_size, square_size, (rvec, tvec)

def highlight_chess_move(
    img,
    move_notation,
    inner_corners,
    board_size,
    square_size,
    pose,
    show_axes=False,
    board=None,
    piece_models=None,
    undistort=True,          # <--- NEW
):
    """
    Highlights chess moves on a perspective view of a chess board.

    Args:
        img: Input image
        move_notation: Chess move in algebraic notation
        inner_corners: Corner points of the inner board
        board_size: Size of the full board
        square_size: Size of each square
        pose: Tuple of (rvec, tvec) for board position
        show_axes: Boolean to control axis display (default False)
        board: Optional ``chess.Board`` representing the current state. If
            provided along with ``piece_models`` the board will be rendered in
            3‑D and composited on top of the highlighted image.
        piece_models: Mapping returned by :func:`load_piece_models` with STL
            meshes for each piece. White pieces are shown in blue and black
            pieces in red.
    """
    # First undistort the image (only if requested and there is nonzero distortion)
    if undistort and np.any(np.abs(dist_coeffs) > 1e-6):
        img = cv2.undistort(img, camera_matrix, dist_coeffs)
        # Keep points consistent with the undistorted image:
        inner_corners = cv2.undistortPoints(
            inner_corners.reshape(-1, 1, 2),
            camera_matrix,
            dist_coeffs,
            P=camera_matrix,
        ).reshape(-1, 2).astype(np.float32)
    
    rvec, tvec = pose
          
    # Calculate destination points with padding for 8x8 board
    padding = square_size  # One square padding on each side
    inner_board_size = 6 * square_size  # Size of the 6x6 inner board
    
    # Define points for the inner board (6x6) with padding
    dst_points_inner = np.float32([
        [padding, padding],  # top-left with padding
        [padding + inner_board_size, padding],  # top-right with padding
        [padding + inner_board_size, padding + inner_board_size],  # bottom-right with padding
        [padding, padding + inner_board_size]  # bottom-left with padding
    ])
    
    # Calculate perspective transform matrix
    matrix = cv2.getPerspectiveTransform(inner_corners, dst_points_inner)
    inv_matrix = cv2.getPerspectiveTransform(dst_points_inner, inner_corners)
    
    # Warp the image to get a top-down view
    warped = cv2.warpPerspective(img, matrix, (board_size, board_size))
    
    # Convert move notation to board coordinates
    file_to_col = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
    
    # Parse move notation
    from_square = (8 - int(move_notation[1]), file_to_col[move_notation[0]])
    to_square = (8 - int(move_notation[3]), file_to_col[move_notation[2]])
    
    # Create overlay for highlighting
    overlay = warped.copy()
    
    # Draw source square (yellow, semi-transparent)
    start_y = from_square[0] * square_size
    start_x = from_square[1] * square_size
    cv2.rectangle(overlay, (start_x, start_y),
                 (start_x + square_size, start_y + square_size),
                 (0, 255, 255), -1)
    
    # Draw destination square (green, semi-transparent)
    end_y = to_square[0] * square_size
    end_x = to_square[1] * square_size
    cv2.rectangle(overlay, (end_x, end_y),
                 (end_x + square_size, end_y + square_size),
                 (0, 255, 0), -1)
    
    # Blend the overlay with the warped image
    alpha = 0.3
    highlighted_warped = cv2.addWeighted(overlay, alpha, warped, 1 - alpha, 0)
    
    # Warp the highlighted image back to the original perspective
    result = cv2.warpPerspective(highlighted_warped, inv_matrix, 
                               (img.shape[1], img.shape[0]))
    
    # Combine the original image with the highlighted overlay
    mask = cv2.warpPerspective(np.ones_like(warped), inv_matrix,
                              (img.shape[1], img.shape[0]))
    final = img.copy()
    final = np.where(mask > 0, result, img)
    
    # Draw coordinate axes only if show_axes is True
    if show_axes:
        axis_length = SQUARE_SIZE / 1000.0  # Convert mm to meters - exactly one square length
        cv2.drawFrameAxes(final, camera_matrix, dist_coeffs, rvec, tvec, axis_length, 3)

    if board is not None and piece_models is not None:
        final = render_board_state(final, board, piece_models, pose, camera_matrix)

    return final



def save_board_transform(camera_id, inner_corners, board_size, square_size, pose, mode, use_white_side):
    """
    Save the board transform data to a JSON file for both cameras
    """
    rvec, tvec = pose

    # Load existing transforms if any
    try:
        with open(BOARD_TRANSFORM_PATH, 'r') as f:
            transform_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        transform_data = {}

    transform_data = _normalize_transform_data(transform_data)

    mode_data = transform_data.setdefault(mode, {})

    # Update transform for current camera
    mode_data[str(camera_id)] = {
        'inner_corners': inner_corners.tolist(),
        'board_size': int(board_size),
        'square_size': int(square_size),
        'rvec': rvec.tolist(),
        'tvec': tvec.tolist(),
        'use_white_side': bool(use_white_side),
        'timestamp': datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    }

    # Save updated transforms
    with open(BOARD_TRANSFORM_PATH, 'w') as f:
        json.dump(transform_data, f, indent=4)

    print(f"Board transform saved for camera {camera_id} in {mode} mode")

# Example usage with images
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chess vision board highlighter")
    parser.add_argument(
        "--mode",
        choices=[REAL_MODE, SIM_MODE],
        default=REAL_MODE,
        help="Select 'real' for physical cameras or 'sim' for Isaac Sim sensors.",
    )
    args = parser.parse_args()

    mode = args.mode

    try:
        if mode == SIM_MODE:
            camera_configs = {
                'white': {
                    'label': 'white',
                    'source': IsaacSimCameraSource(SIM_WHITE_CAMERA),
                    'transform_key': SIM_WHITE_CAMERA,
                    'use_white_side': True,
                },
                'black': {
                    'label': 'black',
                    'source': IsaacSimCameraSource(SIM_BLACK_CAMERA),
                    'transform_key': SIM_BLACK_CAMERA,
                    'use_white_side': False,
                },
            }
        else:
            camera_configs = {
                'white': {
                    'label': 'white',
                    'source': OpenCVCameraSource(REAL_WHITE_CAMERA),
                    'transform_key': str(REAL_WHITE_CAMERA),
                    'use_white_side': True,
                },
                'black': {
                    'label': 'black',
                    'source': OpenCVCameraSource(REAL_BLACK_CAMERA),
                    'transform_key': str(REAL_BLACK_CAMERA),
                    'use_white_side': False,
                },
            }
    except RuntimeError as exc:
        print(exc)
        exit(1)

    transform_lookup = {
        config['transform_key']: config for config in camera_configs.values()
    }

    current_camera_key = 'white'
    cap = camera_configs[current_camera_key]['source']
    if not cap.ensure_ready():
        print(f"Could not open {camera_configs[current_camera_key]['label']} camera in {mode} mode!")
        exit(1)

    print("Camera opened successfully! Press 'q' to quit, 'c' to switch cameras...")

    load_saved_transform(mode, transform_lookup)

    # Initialize variables for storing latest board detection
    latest_corners = None
    latest_board_size = None
    latest_square_size = None
    latest_pose = None

    move = "e2e4"  # Example move

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        transform_key = camera_configs[current_camera_key]['transform_key']
        use_white_side = camera_configs[current_camera_key]['use_white_side']

        try:
            try:
                # Try to detect the board and store the results
                latest_corners, latest_board_size, latest_square_size, latest_pose = \
                    find_chessboard_corners(frame, use_white_side)

                # Use the latest detection for highlighting
                result = highlight_chess_move(
                    frame,
                    move,
                    latest_corners,
                    latest_board_size,
                    latest_square_size,
                    latest_pose,
                )

            except Exception:
                # If detection fails, try using saved transform
                if saved_transform is None or saved_transform.get(transform_key) is None:
                    raise ValueError("No valid transform available")

                camera_transform = saved_transform[transform_key]
                result = highlight_chess_move(
                    frame,
                    move,
                    camera_transform['inner_corners'],
                    camera_transform['board_size'],
                    camera_transform['square_size'],
                    (camera_transform['rvec'], camera_transform['tvec'])
                )

            cv2.imshow("Chess Move Highlight", result)

        except Exception:
            cv2.putText(
                frame,
                "No valid transform available",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            cv2.imshow("Chess Move Highlight", frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Switch cameras
            cap.release()
            new_camera_key = 'black' if current_camera_key == 'white' else 'white'
            new_cap = camera_configs[new_camera_key]['source']

            if new_cap.ensure_ready():
                cap = new_cap
                current_camera_key = new_camera_key

                # Reset latest detection
                latest_corners = None
                latest_board_size = None
                latest_square_size = None
                latest_pose = None

                print(f"Switched to {camera_configs[current_camera_key]['label']} side camera")
            else:
                print(f"Failed to open {camera_configs[new_camera_key]['label']} camera")
                cap = camera_configs[current_camera_key]['source']
                cap.ensure_ready()

        elif key == ord('s'):
            if latest_corners is not None:
                save_board_transform(
                    transform_key,
                    latest_corners,
                    latest_board_size,
                    latest_square_size,
                    latest_pose,
                    mode,
                    camera_configs[current_camera_key]['use_white_side'],
                )
                # Update the stored transform after saving
                load_saved_transform(mode, transform_lookup)
                print(
                    "Transform saved for "
                    f"{camera_configs[current_camera_key]['label']} camera ({transform_key})"
                )
            else:
                print("No valid board detection to save!")

    cap.release()
    cv2.destroyAllWindows()
