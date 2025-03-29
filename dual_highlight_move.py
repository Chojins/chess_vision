import cv2
import pyvirtualcam
import chess_vision
import time
import numpy as np
import subprocess
import chess
import chess.engine
from pyvirtualcam import PixelFormat

def setup_virtual_cameras():
    """
    Creates two virtual cameras: /dev/video4 and /dev/video5.
    Adjust the parameters below if your system needs different device IDs.
    """
    print("Setting up 2 virtual cameras (4 and 5)...")
    # Remove any existing v4l2loopback
    subprocess.run(['sudo', 'modprobe', '-r', 'v4l2loopback'], capture_output=True)
    time.sleep(1)

    # Create 2 virtual devices: /dev/video4 and /dev/video5
    subprocess.run([
        'sudo', 'modprobe', 'v4l2loopback',
        'devices=2',
        'video_nr=4,5',
        'card_label="Virtual Chess Cam 4","Virtual Chess Cam 5"',
        'exclusive_caps=1'
    ])
    time.sleep(1)

    # Permissions so non-root can write to them
    subprocess.run(['sudo', 'chmod', '666', '/dev/video4'])
    subprocess.run(['sudo', 'chmod', '666', '/dev/video5'])

    print("Virtual camera setup complete!")

def setup_chess_engine():
    """Initialize a chess board and a Stockfish engine."""
    board = chess.Board()
    try:
        # Adjust the path to your Stockfish as needed
        engine = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")
        print("Chess engine initialized successfully!")
        return board, engine
    except Exception as e:
        print(f"Error initializing chess engine: {e}")
        return None, None

def get_next_move(board, engine):
    """Ask the engine to find the next move with a brief think time."""
    result = engine.play(board, chess.engine.Limit(time=0.1))
    return result.move

def main():
    # 1) Setup 2 virtual cameras: /dev/video4 and /dev/video5
    setup_virtual_cameras()

    # 2) Initialize chess engine
    board, engine = setup_chess_engine()
    if not board or not engine:
        print("Failed to initialize chess engine!")
        return

    # 3) Open two physical cameras
    #    camera 0 = black side vantage
    #    camera 2 = white side vantage
    cap_black = cv2.VideoCapture(0)  
    cap_white = cv2.VideoCapture(2)  

    if not cap_black.isOpened():
        print("Could not open camera 0 (black side)!")
        engine.quit()
        return

    if not cap_white.isOpened():
        print("Could not open camera 2 (white side)!")
        engine.quit()
        return

    # Set camera properties
    cap_black.set(cv2.CAP_PROP_FPS, 30)
    cap_black.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_black.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cap_white.set(cv2.CAP_PROP_FPS, 30)
    cap_white.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_white.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 4) Load transforms for black and white side from chess_vision
    print("Loading saved transforms...")
    chess_vision.load_saved_transform()
    black_transform = chess_vision.saved_transform.get(chess_vision.BLACK_SIDE_CAMERA, None)
    white_transform = chess_vision.saved_transform.get(chess_vision.WHITE_SIDE_CAMERA, None)
    if not black_transform or not white_transform:
        print("Error: Missing black and/or white transforms in saved_transform!")
        cap_black.release()
        cap_white.release()
        engine.quit()
        return

    # 5) Prepare two virtual cameras: /dev/video4 and /dev/video5
    with pyvirtualcam.Camera(width=640, height=480, fps=30,
                             fmt=PixelFormat.RGB, device='/dev/video4') as virtual_cam4, \
         pyvirtualcam.Camera(width=640, height=480, fps=30,
                             fmt=PixelFormat.RGB, device='/dev/video5') as virtual_cam5:

        print(f'Created virtual cameras: {virtual_cam4.device} and {virtual_cam5.device}')
        print("Press 'q' to quit. Press SPACE for next move. Press 'r' to reset.")

        current_move = None
        # Will store whether the last move was made by White (True) or Black (False).
        # Initially, there is no last move. We'll keep this as None until space is pressed.
        last_move_was_white = None

        while True:
            ret_black, frame_black = cap_black.read()
            ret_white, frame_white = cap_white.read()

            if not ret_black or not ret_white:
                print("Failed to read from black or white camera.")
                break

            # Decide vantage based on who *made* the last move
            if current_move is not None and last_move_was_white is not None:
                # If White just moved => highlight using BLACK vantage
                if last_move_was_white:
                    rvec, tvec = black_transform['rvec'], black_transform['tvec']
                    inner_corners = black_transform['inner_corners']
                    board_size = black_transform['board_size']
                    square_size = black_transform['square_size']

                    move_str = current_move.uci()
                    try:
                        highlighted_black = chess_vision.highlight_chess_move(
                            frame_black,
                            move_str,
                            inner_corners,
                            board_size,
                            square_size,
                            (rvec, tvec),
                            show_axes=False
                        )
                    except Exception as e:
                        print(f"Highlight failed on black side: {e}")
                        highlighted_black = frame_black

                    # Send highlighted black vantage to Virtual Cam 4
                    out_black_rgb = cv2.cvtColor(highlighted_black, cv2.COLOR_BGR2RGB)
                    virtual_cam4.send(out_black_rgb)
                    virtual_cam4.sleep_until_next_frame()

                    # Send the white vantage raw to Virtual Cam 5
                    out_white_rgb = cv2.cvtColor(frame_white, cv2.COLOR_BGR2RGB)
                    virtual_cam5.send(out_white_rgb)
                    virtual_cam5.sleep_until_next_frame()

                    # Preview
                    cv2.imshow("Black side", highlighted_black)
                    cv2.imshow("White side", frame_white)

                else:
                    # Black just moved => highlight using WHITE vantage
                    rvec, tvec = white_transform['rvec'], white_transform['tvec']
                    inner_corners = white_transform['inner_corners']
                    board_size = white_transform['board_size']
                    square_size = white_transform['square_size']

                    move_str = current_move.uci()
                    try:
                        highlighted_white = chess_vision.highlight_chess_move(
                            frame_white,
                            move_str,
                            inner_corners,
                            board_size,
                            square_size,
                            (rvec, tvec),
                            show_axes=False
                        )
                    except Exception as e:
                        print(f"Highlight failed on white side: {e}")
                        highlighted_white = frame_white

                    # Send highlighted white vantage to Virtual Cam 5
                    out_white_rgb = cv2.cvtColor(highlighted_white, cv2.COLOR_BGR2RGB)
                    virtual_cam5.send(out_white_rgb)
                    virtual_cam5.sleep_until_next_frame()

                    # Send the black vantage raw to Virtual Cam 4
                    out_black_rgb = cv2.cvtColor(frame_black, cv2.COLOR_BGR2RGB)
                    virtual_cam4.send(out_black_rgb)
                    virtual_cam4.sleep_until_next_frame()

                    # Preview
                    cv2.imshow("Black side", frame_black)
                    cv2.imshow("White side", highlighted_white)

            else:
                # No move yet => show both cameras raw
                # Black vantage => Virtual Cam 4
                out_black_rgb = cv2.cvtColor(frame_black, cv2.COLOR_BGR2RGB)
                virtual_cam4.send(out_black_rgb)
                virtual_cam4.sleep_until_next_frame()

                # White vantage => Virtual Cam 5
                out_white_rgb = cv2.cvtColor(frame_white, cv2.COLOR_BGR2RGB)
                virtual_cam5.send(out_white_rgb)
                virtual_cam5.sleep_until_next_frame()

                # Preview
                cv2.imshow("Black side", frame_black)
                cv2.imshow("White side", frame_white)

            # Keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                if not board.is_game_over():
                    # BEFORE pushing the move, note who is about to move
                    # True if White is about to move, else False
                    last_move_was_white = board.turn

                    # Engine picks next move
                    next_move = get_next_move(board, engine)
                    print(f"Engine move: {next_move.uci()}")

                    board.push(next_move)
                    current_move = next_move
                else:
                    print("Game is over!")
                    print(f"Result: {board.result()}")
            elif key == ord('r'):
                board.reset()
                current_move = None
                last_move_was_white = None
                print("Board reset.")

    # Cleanup
    cap_black.release()
    cap_white.release()
    cv2.destroyAllWindows()
    engine.quit()

if __name__ == "__main__":
    main()
