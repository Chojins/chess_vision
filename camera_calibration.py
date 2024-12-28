import numpy as np
import cv2
import os
import pickle

# Force X11 backend
os.environ["QT_QPA_PLATFORM"] = "xcb"

def calibrate_camera():
    # Chessboard dimensions
    CHESSBOARD_SIZE = (7, 7)  # Interior points (8x8 board has 7x7 interior corners)
    SQUARE_SIZE = 22.5  # millimeters - adjust this to your board's actual size

    # Prepare object points
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp = objp * SQUARE_SIZE

    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera 0, trying camera 1...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("No cameras available!")
            return False

    # Get frame size
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        return False
    
    frame_size = frame.shape[:2][::-1]
    required_corners = 15
    found_corners = 0
    
    print(f"Starting calibration. Need {required_corners} good frames.")
    print("Press 'c' to capture when the board is detected (shown in green)")
    print("Press 'q' to quit")

    while found_corners < required_corners:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Add image inversion
        gray = cv2.bitwise_not(gray)
        
        # Try different preprocessing methods
        found = False
        corners = None
        
        # Method 1: Original image (now inverted)
        ret, corners = cv2.findChessboardCorners(
            gray, 
            CHESSBOARD_SIZE,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                  cv2.CALIB_CB_NORMALIZE_IMAGE +
                  cv2.CALIB_CB_EXHAUSTIVE
        )
        
        if not ret:
            # Method 2: Try with histogram equalization (on inverted image)
            gray_eq = cv2.equalizeHist(gray)
            ret, corners = cv2.findChessboardCorners(
                gray_eq, 
                CHESSBOARD_SIZE,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                      cv2.CALIB_CB_NORMALIZE_IMAGE +
                      cv2.CALIB_CB_EXHAUSTIVE
            )
            
        if not ret:
            # Method 3: Try with improved adaptive thresholding
            # Add Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            processed = cv2.adaptiveThreshold(
                blurred,  # Use blurred image
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                21,     # Increased block size from 11 to 21
                8       # Increased constant from 2 to 8 to reduce noise in white areas
            )
            
            # Optional: Add morphological operations to clean up the image
            kernel = np.ones((3,3), np.uint8)
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)  # Fill small holes
            processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)   # Remove small noise
            
            ret, corners = cv2.findChessboardCorners(
                processed, 
                CHESSBOARD_SIZE,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                      cv2.CALIB_CB_NORMALIZE_IMAGE +
                      cv2.CALIB_CB_EXHAUSTIVE
            )

        # If found, refine and display
        frame_display = frame.copy()
        if ret:
            # Refine corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Draw corners
            cv2.drawChessboardCorners(frame_display, CHESSBOARD_SIZE, corners, ret)
            
            # Display instructions
            cv2.putText(frame_display, f"Captured: {found_corners}/{required_corners}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame_display, "Press 'c' to capture", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # Simply show "No chessboard detected" on the original frame
            cv2.putText(frame_display, "No chessboard detected", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow('Calibration', frame_display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            found_corners += 1
            print(f"Captured frame {found_corners}/{required_corners}")

    cap.release()
    cv2.destroyAllWindows()

    if found_corners < required_corners:
        print("Calibration incomplete - not enough corners found")
        return False

    print("Calculating calibration...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, frame_size, None, None
    )

    # Calculate reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    print(f"Total reprojection error: {mean_error/len(objpoints)}")

    # Save the calibration data
    calibration_data = {
        'camera_matrix': mtx,
        'dist_coeffs': dist,
        'rvecs': rvecs,
        'tvecs': tvecs,
        'reprojection_error': mean_error/len(objpoints)
    }
    
    with open('camera_calibration.pkl', 'wb') as f:
        pickle.dump(calibration_data, f)
    
    print("Calibration data saved to 'camera_calibration.pkl'")
    return True

if __name__ == "__main__":
    calibrate_camera()