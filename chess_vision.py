import cv2
import numpy as np
import os
import glob

def find_chessboard_corners(img):
    """
    Detect chessboard corners using OpenCV's built-in functions.
    Returns the corners in clockwise order: top-left, top-right, bottom-right, bottom-left
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Invert the image for better detection with dark background
    gray = cv2.bitwise_not(gray)
    
    # Try findChessboardCorners with more relaxed parameters
    ret, corners = cv2.findChessboardCorners(
        gray, 
        (7, 7),  # Changed to 7x7 interior points like chess_simple.py
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
              cv2.CALIB_CB_NORMALIZE_IMAGE +
              cv2.CALIB_CB_EXHAUSTIVE
    )
    
    if not ret:
        # Try with histogram equalization if first attempt fails
        gray_eq = cv2.equalizeHist(gray)
        ret, corners = cv2.findChessboardCorners(
            gray_eq, 
            (7, 7),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                  cv2.CALIB_CB_NORMALIZE_IMAGE +
                  cv2.CALIB_CB_EXHAUSTIVE
        )
    
    if not ret:
        # Show simple "not found" message instead of debug view
        cv2.putText(img, "No chessboard detected", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Chess Move Highlight", img)
        cv2.waitKey(1)
        raise ValueError("Could not detect chessboard corners")
    
    # Refine corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    
    # Get the outer corners for the full 8x8 board
    corner_pts = corners.reshape(7, 7, 2)
    
    # Calculate the average square size
    dx = np.mean(np.diff(corner_pts[:, :, 0], axis=1))
    dy = np.mean(np.diff(corner_pts[:, :, 1], axis=0))
    
    # Calculate outer corners
    top_left = corner_pts[0, 0] - [dx, dy]
    top_right = corner_pts[0, -1] + [dx, -dy]
    bottom_right = corner_pts[-1, -1] + [dx, dy]
    bottom_left = corner_pts[-1, 0] + [-dx, dy]
    
    # Draw outer corners and board outline
    outer_corners = np.float32([top_left, top_right, bottom_right, bottom_left])
    frame_draw = img.copy()
    
    for corner in outer_corners:
        cv2.circle(frame_draw, tuple(corner.astype(int)), 5, (0, 0, 255), -1)
    
    pts = outer_corners.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(frame_draw, [pts], True, (0, 255, 0), 2)
    
    # Display the corners
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
    # Find chess board corners
    board_corners = find_chessboard_corners(img)
    print(f"Found board corners: {board_corners}")
    
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
