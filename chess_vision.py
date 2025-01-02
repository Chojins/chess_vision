import cv2
import numpy as np
import os
import glob
import pickle
import json
import datetime

SQUARE_SIZE = 22.5  # millimeters
CHESSBOARD_SIZE = (7, 7)

# Load camera calibration data
with open('camera_calibration.pkl', 'rb') as f:
    calibration_data = pickle.load(f)
    
camera_matrix = calibration_data['camera_matrix']
dist_coeffs = calibration_data['dist_coeffs']

WHITE_SIDE_CAMERA = 0
BLACK_SIDE_CAMERA = 2

saved_transform = None

def load_saved_transform():
    """
    Load and store the transform data for both cameras
    """
    global saved_transform
    try:
        with open('board_transform.json', 'r') as f:
            data = json.load(f)
        
        # Now expecting a dict with transforms for both cameras
        saved_transform = {
            WHITE_SIDE_CAMERA: {
                'inner_corners': np.array(data[str(WHITE_SIDE_CAMERA)]['inner_corners'], dtype=np.float32),
                'board_size': data[str(WHITE_SIDE_CAMERA)]['board_size'],
                'square_size': data[str(WHITE_SIDE_CAMERA)]['square_size'],
                'rvec': np.array(data[str(WHITE_SIDE_CAMERA)]['rvec']),
                'tvec': np.array(data[str(WHITE_SIDE_CAMERA)]['tvec']),
                'use_white_side': True
            } if str(WHITE_SIDE_CAMERA) in data else None,
            BLACK_SIDE_CAMERA: {
                'inner_corners': np.array(data[str(BLACK_SIDE_CAMERA)]['inner_corners'], dtype=np.float32),
                'board_size': data[str(BLACK_SIDE_CAMERA)]['board_size'],
                'square_size': data[str(BLACK_SIDE_CAMERA)]['square_size'],
                'rvec': np.array(data[str(BLACK_SIDE_CAMERA)]['rvec']),
                'tvec': np.array(data[str(BLACK_SIDE_CAMERA)]['tvec']),
                'use_white_side': False
            } if str(BLACK_SIDE_CAMERA) in data else None
        }
        print("Loaded saved transforms from file")
    except Exception as e:
        print(f"Error loading transform data: {e}")
        saved_transform = {WHITE_SIDE_CAMERA: None, BLACK_SIDE_CAMERA: None}

def find_chessboard_corners(img, use_lowest_corner=True):
    """
    Detect chessboard corners using OpenCV's built-in functions.
    Args:
        img: Input image
        use_lowest_corner: If True, use the white side in the image,
                         if False, use the black side
    """
    # Try finding corners on original image first
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    
    ret, corners = cv2.findChessboardCorners(
        gray, 
        (7, 7),
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
              cv2.CALIB_CB_NORMALIZE_IMAGE +
              cv2.CALIB_CB_EXHAUSTIVE
    )
    
    if not ret:
        # If failed, try with undistorted image
        undistorted = cv2.undistort(img, camera_matrix, dist_coeffs)
        gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        ret, corners = cv2.findChessboardCorners(
            gray, 
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
    square_size = SQUARE_SIZE / 1000.0  # Convert mm to meters
    objp *= square_size
    
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
    if use_lowest_corner:
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
    objp[:, 1] = 6 * square_size - objp[:, 1]
    
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
    
    # If we've fount the incorrect corner, we need to transform to the correct corner
    DISTANCE_THRESHOLD = 250  # mm
    if (use_lowest_corner and distance_mm > DISTANCE_THRESHOLD) or \
       (not use_lowest_corner and distance_mm <= DISTANCE_THRESHOLD):
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

def highlight_chess_move(img, move_notation, use_lowest_corner=True):
    """
    Highlights chess moves on a perspective view of a chess board.
    Args:
        img: Input image
        move_notation: Chess move in algebraic notation
        use_lowest_corner: If True, use the white side in the image,
                         if False, use the black side
    """
    global saved_transform
    
    # First undistort the image
    img = cv2.undistort(img, camera_matrix, dist_coeffs)
    
    try:
        # Try to detect the board
        inner_corners, board_size, square_size, pose = find_chessboard_corners(img, use_lowest_corner)
        rvec, tvec = pose
    except Exception as e:
        # If detection fails, use stored transform for current camera
        if saved_transform is None or saved_transform[current_camera] is None:
            raise ValueError("No valid transform available")
            
        # Get transform for current camera
        camera_transform = saved_transform[current_camera]
        inner_corners = camera_transform['inner_corners']
        board_size = camera_transform['board_size']
        square_size = camera_transform['square_size']
        rvec = camera_transform['rvec']
        tvec = camera_transform['tvec']
        
        # Check if we need to transform based on distance threshold
        distance_mm = np.linalg.norm(tvec) * 1000
        DISTANCE_THRESHOLD = 250  # mm
        needs_transform = (use_white_side and distance_mm > DISTANCE_THRESHOLD) or \
                        (not use_white_side and distance_mm <= DISTANCE_THRESHOLD)
        
        # Also transform if the saved perspective doesn't match current
        if needs_transform != (camera_transform['use_white_side'] != use_white_side):
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
            
            # Transform the inner corners as well
            corners_homog = np.ones((4, 3))
            corners_homog[:, :2] = inner_corners
            
            # Apply 180° rotation and translation to corners
            for i in range(4):
                corners_homog[i, :2] = board_size - corners_homog[i, :2]
            
            # Update inner_corners
            inner_corners = corners_homog[:, :2]
    
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
    
    # Draw coordinate axes on final image with shorter length
    axis_length = SQUARE_SIZE / 1000.0  # Convert mm to meters - exactly one square length
    cv2.drawFrameAxes(final, camera_matrix, dist_coeffs, rvec, tvec, axis_length, 3)
    
    return final

def save_board_transform(camera_id, inner_corners, board_size, square_size, pose):
    """
    Save the board transform data to a JSON file for both cameras
    """
    rvec, tvec = pose
    
    # Load existing transforms if any
    try:
        with open('board_transform.json', 'r') as f:
            transform_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        transform_data = {}
    
    # Update transform for current camera
    transform_data[str(camera_id)] = {
        'inner_corners': inner_corners.tolist(),
        'board_size': int(board_size),
        'square_size': int(square_size),
        'rvec': rvec.tolist(),
        'tvec': tvec.tolist(),
        'timestamp': datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    # Save updated transforms
    with open('board_transform.json', 'w') as f:
        json.dump(transform_data, f, indent=4)
    
    print(f"Board transform saved for camera {camera_id}")

# Example usage with images
if __name__ == "__main__":
    # Initialize with white side camera
    current_camera = WHITE_SIDE_CAMERA
    cap = cv2.VideoCapture(current_camera)
    if not cap.isOpened():
        print(f"Could not open camera {current_camera}!")
        exit(1)

    # Set camera properties
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Camera opened successfully! Press 'q' to quit, 'c' to switch cameras...")
    
    # Load the saved transforms at startup
    load_saved_transform()
    
    move = "e2e4"  # Example move
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        try:
            # use_white_side should match the current camera
            use_white_side = (current_camera == WHITE_SIDE_CAMERA)
            result = highlight_chess_move(frame, move, use_white_side)
            cv2.imshow("Chess Move Highlight", result)
            
            # Try to detect the board for saving purposes
            try:
                inner_corners, board_size, square_size, pose = find_chessboard_corners(frame, use_white_side)
                latest_corners = inner_corners
                latest_board_size = board_size
                latest_square_size = square_size
                latest_pose = pose
            except Exception:
                pass
            
        except Exception as e:
            cv2.putText(frame, "No valid transform available", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Chess Move Highlight", frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Switch cameras
            new_camera = BLACK_SIDE_CAMERA if current_camera == WHITE_SIDE_CAMERA else WHITE_SIDE_CAMERA
            new_cap = cv2.VideoCapture(new_camera)
            
            if new_cap.isOpened():
                # Close the old camera
                cap.release()
                
                # Set up the new camera
                cap = new_cap
                current_camera = new_camera
                
                # Set camera properties
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                print(f"Switched to {'black' if current_camera == BLACK_SIDE_CAMERA else 'white'} side camera")
            else:
                print(f"Failed to open camera {new_camera}")
                new_cap.release()
                
        elif key == ord('s'):
            if latest_corners is not None:
                save_board_transform(current_camera, latest_corners, latest_board_size, 
                                  latest_square_size, latest_pose)
                # Update the stored transform after saving
                load_saved_transform()
            else:
                print("No valid board detection to save!")
    
    cap.release()
    cv2.destroyAllWindows()
