import cv2
import chess_vision
import time

def main():
    # Initialize with white side camera
    chess_vision.current_camera = chess_vision.WHITE_SIDE_CAMERA
    cap = cv2.VideoCapture(chess_vision.current_camera)
    if not cap.isOpened():
        print(f"Could not open camera {chess_vision.current_camera}!")
        exit(1)

    # Set camera properties
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Load the saved transforms at startup
    chess_vision.load_saved_transform()
    
    # Example move - you can change this or make it interactive
    move = "e2e4"
    
    print("Press 'q' to quit, 'c' to switch cameras, 'm' to input a new move...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        try:
            # use_white_side should match the current camera
            use_white_side = (chess_vision.current_camera == chess_vision.WHITE_SIDE_CAMERA)
            
            # Get transform for current camera
            if chess_vision.saved_transform is None or chess_vision.saved_transform[chess_vision.current_camera] is None:
                raise ValueError("No valid transform available")
                
            camera_transform = chess_vision.saved_transform[chess_vision.current_camera]
            inner_corners = camera_transform['inner_corners']
            board_size = camera_transform['board_size']
            square_size = camera_transform['square_size']
            pose = (camera_transform['rvec'], camera_transform['tvec'])
            
            # Pass the transform data directly to highlight_chess_move
            result = chess_vision.highlight_chess_move(frame, move, inner_corners, board_size, square_size, pose)
            cv2.imshow("Chess Move Highlight", result)
            
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
            new_camera = chess_vision.BLACK_SIDE_CAMERA if chess_vision.current_camera == chess_vision.WHITE_SIDE_CAMERA else chess_vision.WHITE_SIDE_CAMERA
            new_cap = cv2.VideoCapture(new_camera)
            
            if new_cap.isOpened():
                cap.release()
                cap = new_cap
                chess_vision.current_camera = new_camera
                
                # Set camera properties
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                print(f"Switched to {'black' if chess_vision.current_camera == chess_vision.BLACK_SIDE_CAMERA else 'white'} side camera")
            else:
                print(f"Failed to open camera {new_camera}")
                new_cap.release()
        elif key == ord('m'):
            # Get new move from user
            print("\nEnter move (e.g., e2e4):", end=' ')
            move = input().strip()
            print(f"New move set to: {move}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()