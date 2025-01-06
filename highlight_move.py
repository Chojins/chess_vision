import cv2
import pyvirtualcam
import chess_vision
import time
import numpy as np
import subprocess
import chess
import chess.engine

def setup_virtual_camera():
    print("Setting up virtual camera...")
    subprocess.run(['sudo', 'modprobe', '-r', 'v4l2loopback'], capture_output=True)
    time.sleep(1)
    subprocess.run([
        'sudo', 'modprobe', 'v4l2loopback',
        'devices=1',
        'video_nr=4',
        'card_label="Virtual Chess Camera"',
        'exclusive_caps=1'
    ])
    time.sleep(1)
    subprocess.run(['sudo', 'chmod', '666', '/dev/video4'])
    print("Virtual camera setup complete!")

def setup_chess_engine():
    # Initialize chess board and engine
    board = chess.Board()
    try:
        # Adjust the path to where stockfish is installed on your system
        engine = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")
        print("Chess engine initialized successfully!")
        return board, engine
    except Exception as e:
        print(f"Error initializing chess engine: {e}")
        return None, None

def get_next_move(board, engine):
    # Get engine's move with 100ms of thinking time
    result = engine.play(board, chess.engine.Limit(time=0.1))
    return result.move

def main():
    # Setup virtual camera
    setup_virtual_camera()

    # Setup chess engine
    board, engine = setup_chess_engine()
    if not board or not engine:
        print("Failed to initialize chess engine!")
        return

    # Initialize camera 0 (black side camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera 0!")
        engine.quit()
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
        
        # Initialize move state
        current_move = None
        print("Press 'q' to quit, SPACE to get next move...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            try:
                # Get transform for black side camera
                if chess_vision.saved_transform is None or chess_vision.saved_transform[chess_vision.BLACK_SIDE_CAMERA] is None:
                    raise ValueError("No valid transform available for black side camera")
                    
                camera_transform = chess_vision.saved_transform[chess_vision.BLACK_SIDE_CAMERA]
                inner_corners = camera_transform['inner_corners']
                board_size = camera_transform['board_size']
                square_size = camera_transform['square_size']
                pose = (camera_transform['rvec'], camera_transform['tvec'])
                
                # If we have a current move, highlight it
                if current_move:
                    move_str = current_move.uci()
                    result = chess_vision.highlight_chess_move(frame, move_str, 
                                                            inner_corners, board_size, 
                                                            square_size, pose, show_axes=False)
                else:
                    result = frame
                
                # Convert BGR to RGB for virtual camera
                result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                virtual_cam.send(result_rgb)
                
                # Show preview window (removed position text)
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
            elif key == ord(' '):  # Spacebar
                if not board.is_game_over():
                    # Get and make next move
                    current_move = get_next_move(board, engine)
                    print(f"Next move: {current_move.uci()}")
                    board.push(current_move)
                else:
                    print("Game is over!")
                    print(f"Result: {board.result()}")
            elif key == ord('r'):  # Reset game
                board.reset()
                current_move = None
                print("Game reset!")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    engine.quit()

if __name__ == "__main__":
    main()