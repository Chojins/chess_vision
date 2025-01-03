import cv2
import pyvirtualcam
import chess_vision
import time
import numpy as np
import subprocess

def setup_virtual_camera():
    print("Setting up virtual camera...")
    
    # Force remove the module
    subprocess.run(['sudo', 'modprobe', '-r', 'v4l2loopback'], capture_output=True)
    time.sleep(1)
    
    # Create new virtual camera
    subprocess.run([
        'sudo', 'modprobe', 'v4l2loopback',
        'devices=1',
        'video_nr=4',
        'card_label="Virtual Chess Camera"',
        'exclusive_caps=1'
    ])
    time.sleep(1)
    
    # Set permissions
    subprocess.run(['sudo', 'chmod', '666', '/dev/video4'])
    
    print("Virtual camera setup complete!")

def main():
    # Setup virtual camera first
    setup_virtual_camera()

    # Initialize camera 0 (black side camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera 0!")
        exit(1)

    # Set camera properties
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Initialize virtual camera with RGB format
    with pyvirtualcam.Camera(width=640, height=480, fps=30, fmt=pyvirtualcam.PixelFormat.RGB) as virtual_cam:
        print(f'Virtual camera created: {virtual_cam.device}')
     
        # Load the saved transforms at startup
        chess_vision.load_saved_transform()
        
        # Example move - you can change this or make it interactive
        move = "e2e4"
        
        print("Press 'q' to quit, 'm' to input a new move...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            try:
                # Get transform for black side camera
                if chess_vision.saved_transform is None or chess_vision.saved_transform[chess_vision.BLACK_SIDE_CAMERA] is None:
                    raise ValueError("No valid transform available for black side camera")
                    
                camera_transform = chess_vision.saved_transform[chess_vision.BLACK_SIDE_CAMERA]  # Use black side camera's transform
                inner_corners = camera_transform['inner_corners']
                board_size = camera_transform['board_size']
                square_size = camera_transform['square_size']
                pose = (camera_transform['rvec'], camera_transform['tvec'])
                
                # Pass the transform data directly to highlight_chess_move
                result = chess_vision.highlight_chess_move(frame, move, inner_corners, board_size, square_size, pose)
                
                # Convert BGR to RGB for virtual camera
                result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                virtual_cam.send(result_rgb)
                
                # Still show preview window
                cv2.imshow("Chess Move Highlight", result)
                
            except Exception as e:
                # Show error frame
                cv2.putText(frame, "No valid transform available for black side camera", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                virtual_cam.send(frame_rgb)
                cv2.imshow("Chess Move Highlight", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                # Get new move from user
                print("\nEnter move (e.g., e2e4):", end=' ')
                move = input().strip()
                print(f"New move set to: {move}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()