import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import numpy as np
import pickle

# Initialize video capture (try both cameras)
cap = cv2.VideoCapture(2)


# Chessboard parameters
CHESSBOARD_SIZE = (7, 7)
SQUARE_SIZE = 22.5 # millimeters

# Termination criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

with open('camera_calibration.pkl', 'rb') as f:
    calibration_data = pickle.load(f)
    
camera_matrix = calibration_data['camera_matrix']
dist_coeffs = calibration_data['dist_coeffs']

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Invert the image - this is to allow for the black boarder on teh board - The stock opencv 
    #checkerboard patterns have white boarders
    gray = cv2.bitwise_not(gray)

    # Find the chessboard corners with more relaxed parameters
    ret, corners = cv2.findChessboardCorners(
        gray, 
        CHESSBOARD_SIZE,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
              cv2.CALIB_CB_NORMALIZE_IMAGE +
              cv2.CALIB_CB_EXHAUSTIVE
    )

    if not ret:
        # Try with different preprocessing if first attempt fails
        gray_eq = cv2.equalizeHist(gray)  # Enhance contrast
        ret, corners = cv2.findChessboardCorners(
            gray_eq, 
            CHESSBOARD_SIZE,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                  cv2.CALIB_CB_NORMALIZE_IMAGE +
                  cv2.CALIB_CB_EXHAUSTIVE
        )

    # If found, refine and draw
    if ret:
        # Refine corners to sub-pixel accuracy
        corners2 = cv2.cornerSubPix(
            gray, 
            corners, 
            (11, 11),
            (-1, -1), 
            criteria
        )

        # Draw the corners
        frame_draw = frame.copy()
        cv2.drawChessboardCorners(frame_draw, CHESSBOARD_SIZE, corners2, ret)

        # Get the outer corners for the full 8x8 board
        corner_pts = corners2.reshape(7, 7, 2)
        
        # Calculate the average square size
        dx = np.mean(np.diff(corner_pts[:, :, 0], axis=1))
        dy = np.mean(np.diff(corner_pts[:, :, 1], axis=0))
        
        # Calculate outer corners
        top_left = corner_pts[0, 0] - [dx, dy]
        top_right = corner_pts[0, -1] + [dx, -dy]
        bottom_right = corner_pts[-1, -1] + [dx, dy]
        bottom_left = corner_pts[-1, 0] + [-dx, dy]
        
        # Draw outer corners and full board outline
        outer_corners = np.float32([top_left, top_right, bottom_right, bottom_left])
        for corner in outer_corners:
            cv2.circle(frame_draw, tuple(corner.astype(int)), 5, (0, 0, 255), -1)
        
        pts = outer_corners.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(frame_draw, [pts], True, (0, 255, 0), 2)

        # Perspective correction
        # Define the desired size of the board in pixels
        board_size = 400
        dst_points = np.array([
            [0, 0],
            [board_size, 0],
            [board_size, board_size],
            [0, board_size]
        ], dtype=np.float32)
        
        # Calculate perspective transform
        matrix = cv2.getPerspectiveTransform(outer_corners, dst_points)
        warped = cv2.warpPerspective(frame, matrix, (board_size, board_size))
        
        # Show both the original detection and corrected view
        combined_view = np.hstack((frame_draw, cv2.resize(warped, (frame_draw.shape[1], frame_draw.shape[0]))))
        cv2.imshow('Chessboard Detection', combined_view)

    else:
        # Simply show "No chessboard detected" on the original frame
        cv2.putText(frame, "No chessboard detected", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Chessboard Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()