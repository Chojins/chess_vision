import cv2
import numpy as np

def test_cameras():
    # Open both cameras with V4L2 backend
    cap_real = cv2.VideoCapture(2, cv2.CAP_V4L2)
    cap_virtual = cv2.VideoCapture(4, cv2.CAP_V4L2)
    
    # Check if cameras opened successfully
    if not cap_real.isOpened() or not cap_virtual.isOpened():
        print("Error: Could not open one or both cameras")
        return
    
    print("Press 'q' to quit")
    
    while True:
        # Read frames from both cameras
        ret_real, frame_real = cap_real.read()
        ret_virtual, frame_virtual = cap_virtual.read()
        
        # Create side-by-side display
        if ret_real and ret_virtual:
            combined_frame = np.hstack((frame_real, frame_virtual))
            cv2.putText(combined_frame, "Camera 2", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(combined_frame, "Virtual Camera 4", (frame_real.shape[1] + 10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Camera Feeds', combined_frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap_real.release()
    cap_virtual.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_cameras()