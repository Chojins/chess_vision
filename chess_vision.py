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
        # [corners for square], y-coordinate of bottom corner
        (np.array([
            corner_pts[-1, 0],     # bottom-left corner
            corner_pts[-1, 1],     # bottom-right corner
            corner_pts[-2, 1],     # top-right corner
            corner_pts[-2, 0]      # top-left corner
        ]), corner_pts[-1, 0][1]),  # y-coord of bottom-left corner
        
        (np.array([
            corner_pts[-1, -2],    # rotated 90° clockwise
            corner_pts[-1, -1],
            corner_pts[-2, -1],
            corner_pts[-2, -2]
        ]), corner_pts[-1, -1][1]),
        
        (np.array([
            corner_pts[1, -1],     # rotated 180°
            corner_pts[1, -2],
            corner_pts[0, -2],
            corner_pts[0, -1]
        ]), corner_pts[0, -1][1]),
        
        (np.array([
            corner_pts[1, 0],      # rotated 270°
            corner_pts[1, 1],
            corner_pts[0, 1],
            corner_pts[0, 0]
        ]), corner_pts[0, 0][1])
    ]
    
    # Find black corner squares
    black_corners = []
    for i, (square_corners, y_coord) in enumerate(corner_squares):
        color = get_square_color(gray_img, square_corners)
        if color < 128:  # if square is black
            black_corners.append((i, y_coord))
    
    if not black_corners:
        raise ValueError("No black corner squares found!")
    
    # Choose the black corner that's lowest in the image (largest y-coordinate)
    rotations_needed = black_corners[np.argmax([y for _, y in black_corners])][0]
    
    # Rotate the corner_pts array to put the chosen black square at bottom-left
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
    cv2.imshow("Detected Corners", frame_draw)
    
    return inner_corners, board_full_size, square_size

def highlight_chess_move(img, move_notation):
    """
    Highlights chess moves on a perspective view of a chess board.
    """
    # First undistort the image
    img = cv2.undistort(img, camera_matrix, dist_coeffs)
    
    # Find chess board corners
    inner_corners, board_size, square_size = find_chessboard_corners(img)
    
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
    
    # Warp the image to get a top-down view (use full board_size for output)
    warped = cv2.warpPerspective(img, matrix, (board_size, board_size))
    cv2.imshow("Warped Board", warped)
    
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
    
    # Draw debug grid
    debug_overlay = overlay.copy()
    for i in range(9):  # 9 lines for 8 squares
        x = i * square_size
        y = i * square_size
        cv2.line(debug_overlay, (x, 0), (x, board_size), (0, 255, 0), 1)
        cv2.line(debug_overlay, (0, y), (board_size, y), (0, 255, 0), 1)
    cv2.imshow("Debug Grid", debug_overlay)
    
    # Blend the overlay with the warped image
    alpha = 0.3
    highlighted_warped = cv2.addWeighted(overlay, alpha, warped, 1 - alpha, 0)
    cv2.imshow("Highlighted Warped", highlighted_warped)
    
    # Warp the highlighted image back to the original perspective
    result = cv2.warpPerspective(highlighted_warped, inv_matrix, 
                               (img.shape[1], img.shape[0]))
    
    # Combine the original image with the highlighted overlay
    mask = cv2.warpPerspective(np.ones_like(warped), inv_matrix,
                              (img.shape[1], img.shape[0]))
    final = img.copy()
    final = np.where(mask > 0, result, img)
    
    return final

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
