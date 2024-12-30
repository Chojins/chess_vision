import cv2
import numpy as np
import os
import glob
import pickle

# Add these constants at the top of the file
SQUARE_SIZE = 22.5  # millimeters
CHESSBOARD_SIZE = (7, 7)

# Load camera calibration data
with open('camera_calibration.pkl', 'rb') as f:
    calibration_data = pickle.load(f)
    
camera_matrix = calibration_data['camera_matrix']
dist_coeffs = calibration_data['dist_coeffs']

def find_chessboard_corners(img):
    """
    Detect chessboard corners using OpenCV's built-in functions.
    Returns the corners in clockwise order: top-left, top-right, bottom-right, bottom-left
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
    
    # Reshape corners into 7x7 grid
    corner_pts = corners.reshape(7, 7, 2)
    
    # Function to get average color of a square
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
        
        # Debug visualization
        debug_img = img.copy()
        cv2.circle(debug_img, (center_x, center_y), 3, (0, 0, 255), -1)
        cv2.imshow("Color Sampling Debug", debug_img)
        
        return np.mean(roi)

    # Get the corners for the first square (current bottom-left)
    square_corners = np.array([
        corner_pts[-1, 0],     # bottom-left corner
        corner_pts[-1, 1],     # bottom-right corner
        corner_pts[-2, 1],     # top-right corner
        corner_pts[-2, 0]      # top-left corner
    ])
    
    # Convert image to grayscale if it isn't already
    if len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img
    
    # Get color of current bottom-left square
    current_color = get_square_color(gray_img, square_corners)
    print(f"Bottom-left square color value: {current_color}")  # Debug print
    
    # Determine if we need to rotate based on square color
    # Note: in grayscale, lower values are darker (black)
    is_black = current_color < 128  # Assuming middle gray as threshold
    
    # Try all four rotations until we get a black square in bottom-left
    rotations = 0
    while not is_black and rotations < 4:
        corner_pts = np.rot90(corner_pts)
        rotations += 1
        
        # Update square corners and check color again
        square_corners = np.array([
            corner_pts[-1, 0],
            corner_pts[-1, 1],
            corner_pts[-2, 1],
            corner_pts[-2, 0]
        ])
        current_color = get_square_color(gray_img, square_corners)
        print(f"Rotation {rotations}, color value: {current_color}")  # Debug print
        is_black = current_color < 128
    
    if rotations == 4:
        print("Warning: Could not determine correct board orientation based on square colors")
    
    # Reshape corners back to original format
    corners = corner_pts.reshape(-1, 1, 2)
    
    # Now proceed with the rest of the corner processing
    board_inner_size = 700
    square_size = board_inner_size // 7
    
    # Define points for the inner board (now guaranteed to be in correct orientation)
    dst_points_inner = np.array([
        [0, 0],
        [board_inner_size, 0],
        [board_inner_size, board_inner_size],
        [0, board_inner_size]
    ], dtype=np.float32)
    
    # Get corners of detected inner board (now in correct orientation)
    inner_corners = np.float32([
        corner_pts[0, 0],      # top-left
        corner_pts[0, -1],     # top-right
        corner_pts[-1, -1],    # bottom-right
        corner_pts[-1, 0]      # bottom-left
    ])
    
    # Warp the inner board
    matrix = cv2.getPerspectiveTransform(inner_corners, dst_points_inner)
    warped = cv2.warpPerspective(img, matrix, (board_inner_size, board_inner_size))
    
    # Create a larger image with one square padding on all sides
    board_full_size = board_inner_size + (2 * square_size)  # Add two squares (one on each side)
    warped_padded = np.zeros((board_full_size, board_full_size, 3), dtype=np.uint8)
    
    # Copy the warped board into the center of the padded image
    warped_padded[square_size:square_size+board_inner_size, 
                 square_size:square_size+board_inner_size] = warped
    
    # Define the corners for the full board in the warped space
    outer_corners_warped = np.float32([
        [0, 0],  # top-left
        [board_full_size, 0],  # top-right
        [board_full_size, board_full_size],  # bottom-right
        [0, board_full_size]  # bottom-left
    ])
    
    # Calculate the inverse transform to go back to the original perspective
    inv_matrix = cv2.getPerspectiveTransform(outer_corners_warped, inner_corners)
    
    # Get the outer corners in the original perspective
    outer_corners = cv2.perspectiveTransform(
        outer_corners_warped.reshape(-1, 1, 2), 
        inv_matrix
    ).reshape(-1, 2)
    
    # Draw debug visualization if needed
    frame_draw = img.copy()
    for corner in outer_corners:
        cv2.circle(frame_draw, tuple(corner.astype(int)), 5, (0, 0, 255), -1)
    pts = outer_corners.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(frame_draw, [pts], True, (0, 255, 0), 2)
    cv2.imshow("Detected Corners", frame_draw)
    cv2.waitKey(1)
    
    return outer_corners

def highlight_chess_move(img, move_notation):
    """
    Highlights chess moves on a perspective view of a chess board.
    
    Args:
        img: Input image/frame (numpy array)
        move_notation (str): Chess move in standard notation (e.g., 'e2e4')
    
    Returns:
        numpy.ndarray: Image with highlighted squares
    """
    # First undistort the image
    img = cv2.undistort(img, camera_matrix, dist_coeffs)
    
    # Find chess board corners
    board_corners = find_chessboard_corners(img)
    print(f"Found board corners: {board_corners}")
    
    # Calculate the physical size of the full board in mm
    board_physical_size = SQUARE_SIZE * 8  # 8 squares * 22.5mm = 180mm
    
    # Define destination points for perspective transform (square board)
    board_size = 800  # pixels
    dst_points = np.float32([[0, 0], [board_size, 0],
                            [board_size, board_size], [0, board_size]])
    
    # Calculate perspective transform matrix
    matrix = cv2.getPerspectiveTransform(board_corners, dst_points)
    inv_matrix = cv2.getPerspectiveTransform(dst_points, board_corners)
    
    # Warp the image to get a top-down view
    warped = cv2.warpPerspective(img, matrix, (board_size, board_size))
    cv2.imshow("Warped Board", warped)  # Debug: show warped image
    
    # Convert move notation to board coordinates
    file_to_col = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
    
    # Parse move notation
    from_square = (8 - int(move_notation[1]), file_to_col[move_notation[0]])
    to_square = (8 - int(move_notation[3]), file_to_col[move_notation[2]])
    print(f"Move from {from_square} to {to_square}")
    
    # Square size in the warped image
    square_size = board_size // 8
    
    # Create overlay for highlighting
    overlay = warped.copy()
    
    # Draw source square (yellow, semi-transparent)
    start_y = from_square[0] * square_size
    start_x = from_square[1] * square_size
    cv2.rectangle(overlay, (start_x, start_y),
                 (start_x + square_size, start_y + square_size),
                 (0, 255, 255), -1)
    print(f"Drawing source square at ({start_x}, {start_y})")
    
    # Draw destination square (green, semi-transparent)
    end_y = to_square[0] * square_size
    end_x = to_square[1] * square_size
    cv2.rectangle(overlay, (end_x, end_y),
                 (end_x + square_size, end_y + square_size),
                 (0, 255, 0), -1)
    print(f"Drawing destination square at ({end_x}, {end_y})")
    
    # Blend the overlay with the warped image
    alpha = 0.3
    highlighted_warped = cv2.addWeighted(overlay, alpha, warped, 1 - alpha, 0)
    cv2.imshow("Highlighted Warped", highlighted_warped)  # Debug: show highlighted warped image
    
    # Warp the highlighted image back to the original perspective
    result = cv2.warpPerspective(highlighted_warped, inv_matrix, 
                               (img.shape[1], img.shape[0]))
    
    # Combine the original image with the highlighted overlay
    mask = cv2.warpPerspective(np.ones_like(warped), inv_matrix,
                              (img.shape[1], img.shape[0]))
    result = img.copy()
    result = np.where(mask > 0, result, img)
    
    return result

# Example usage with images
if __name__ == "__main__":
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera 0, trying camera 1...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("No cameras available!")
            exit(1)

    print("Camera opened successfully! Press 'q' to quit...")
    
    move = "e2e4"  # Example move
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        try:
            result = highlight_chess_move(frame, move)
            cv2.imshow("Chess Move Highlight", result)
            
            # Break loop with 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        except Exception as e:
            cv2.putText(frame, "No chessboard detected", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Chess Move Highlight", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
