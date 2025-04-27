import cv2
import pyvirtualcam
import chess_vision
import time
import numpy as np
import subprocess
import chess
import chess.engine
import sys
import argparse
from pyvirtualcam import PixelFormat
import os

# --------------------------
# 1) Mask-based Flip Helpers
# --------------------------
def load_flip_mask(mask_path: str, width: int, height: int) -> np.ndarray:
    """
    Loads a black & white mask, converts it to 0-1 float, and
    ensures it matches the (width x height).
    Returns a 3-channel float mask in shape (H x W x 3).
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Error: Could not read mask at {mask_path}")

    # Resize mask if needed
    if (mask.shape[1] != width) or (mask.shape[0] != height):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    mask_float = mask.astype(np.float32) / 255.0
    # Expand to 3 channels
    mask_float_3d = np.dstack([mask_float]*3)
    return mask_float_3d

def flip_frame_with_mask(frame: np.ndarray, mask_float_3d: np.ndarray) -> np.ndarray:
    """
    Flip only the portion of 'frame' where mask_float_3d == 1.
    """
    # Flip the entire frame horizontally
    flipped_frame = cv2.flip(frame, 1)

    # The “mirror-only” portion (where mask = 1)
    mirror_only_flipped = flipped_frame * mask_float_3d

    # The “non-mirror” portion (where mask = 0)
    non_flipped_region = frame * (1.0 - mask_float_3d)

    # Combine them
    combined = mirror_only_flipped + non_flipped_region
    combined_uint8 = combined.astype(np.uint8)

    return combined_uint8


def setup_virtual_cameras():
    """
    Creates FOUR virtual cameras:
        • /dev/video4  (black - LeRobot)
        • /dev/video5  (white - LeRobot)
        • /dev/video6  (black - OBS)
        • /dev/video7  (white - OBS)
    """
    print("Setting up 4 virtual cameras (4-7)…")
    subprocess.run(['sudo', 'modprobe', '-r', 'v4l2loopback'], check=True)
    time.sleep(1)

    subprocess.run([
        'sudo', 'modprobe', 'v4l2loopback',
        'devices=4',
        'video_nr=4,5,6,7',
        ('card_label="SO100 Black","SO100 White",'
         '"OBS Black","OBS White"'),
        'yuv420=1',
        'exclusive_caps=0'          # readers & writers can coexist
    ], check=True)
    time.sleep(1)

    for dev in ('/dev/video4','/dev/video5','/dev/video6','/dev/video7'):
        subprocess.run(['sudo', 'chmod', '666', dev], check=True)

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
    # ----------------------
    # Parse command-line args
    # ----------------------
    parser = argparse.ArgumentParser(
        description="Highlight chess moves with two vantage cameras (black & white)."
    )
    parser.add_argument(
        "--vantage",
        choices=["black", "white"],
        default=None,
        help="Force highlighting from one vantage (black/white). Omit for normal swapping mode."
    )
    parser.add_argument(
        "--flip_mask",
        type=str,
        default=None,
        help="Path to a black-and-white mask image (white=flip, black=keep). "
             "If provided, frames are flipped after highlighting."
    )
    args = parser.parse_args()

    forced_vantage = args.vantage  # None => normal swapping
    flip_mask_path = args.flip_mask

    # 1) Setup 2 virtual cameras: /dev/video4 and /dev/video5
    setup_virtual_cameras()

    # 2) Initialize chess engine
    board, engine = setup_chess_engine()
    if not board or not engine:
        print("Failed to initialize chess engine!")
        return

    # 3) Open two physical cameras
    #    camera 0 => black vantage
    #    camera 2 => white vantage
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

    # 5) (Optional) Load flip mask if provided
    mask_float_3d = None
    if flip_mask_path is not None and os.path.isfile(flip_mask_path):
        try:
            # We'll assume both cameras are 640×480, so:
            mask_float_3d = load_flip_mask(flip_mask_path, 640, 480)
            print(f"Loaded flip mask from {flip_mask_path}")
        except Exception as e:
            print(f"Failed to load/resize flip mask: {e}")

    # 6) Prepare two virtual cameras: /dev/video4 and /dev/video5
    with pyvirtualcam.Camera(640, 480, 30, fmt=PixelFormat.RGB,
                         device='/dev/video4') as v_cam4, \
        pyvirtualcam.Camera(640, 480, 30, fmt=PixelFormat.RGB,
                         device='/dev/video5') as v_cam5, \
        pyvirtualcam.Camera(640, 480, 30, fmt=PixelFormat.RGB,
                         device='/dev/video6') as v_cam6, \
        pyvirtualcam.Camera(640, 480, 30, fmt=PixelFormat.RGB,
                         device='/dev/video7') as v_cam7:

        print(f'LeRobot cams  : {v_cam4.device}, {v_cam5.device}')
        print(f'OBS cams      : {v_cam6.device}, {v_cam7.device}')

        if forced_vantage is None:
            print("MODE: Normal swapping. White moves => black vantage, Black moves => white vantage.")
        else:
            print(f"MODE: Fixed vantage = {forced_vantage} camera for all moves.")

        print("Press 'q' to quit. Press SPACE for next move. Press 'r' to reset the board.")

        current_move = None
        # Tracks which color made the last move (True = White, False = Black)
        last_move_was_white = None

        while True:
            ret_black, frame_black = cap_black.read()
            ret_white, frame_white = cap_white.read()
            if not ret_black or not ret_white:
                print("Failed to read from black or white camera.")
                break

            # We'll define default frames to display
            final_black_frame = frame_black
            final_white_frame = frame_white

            if not current_move or last_move_was_white is None:
                # No move yet => no highlight
                pass
            else:
                # We have a move to highlight
                move_str = current_move.uci()

                if forced_vantage is None:
                    # ---------------------------------------------------
                    # NORMAL (SWAPPING) MODE
                    # ---------------------------------------------------
                    if last_move_was_white:
                        # White just moved => highlight from black vantage
                        rvec, tvec = black_transform['rvec'], black_transform['tvec']
                        inner_corners = black_transform['inner_corners']
                        board_size = black_transform['board_size']
                        square_size = black_transform['square_size']

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

                        final_black_frame = highlighted_black  # final for black vantage

                    else:
                        # Black just moved => highlight from white vantage
                        rvec, tvec = white_transform['rvec'], white_transform['tvec']
                        inner_corners = white_transform['inner_corners']
                        board_size = white_transform['board_size']
                        square_size = white_transform['square_size']

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

                        final_white_frame = highlighted_white

                else:
                    # ---------------------------------------------------
                    # FIXED VANTAGE MODE (forced_vantage = 'black' or 'white')
                    # ---------------------------------------------------
                    if forced_vantage == "black":
                        # Always highlight from black vantage
                        rvec, tvec = black_transform['rvec'], black_transform['tvec']
                        inner_corners = black_transform['inner_corners']
                        board_size = black_transform['board_size']
                        square_size = black_transform['square_size']

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
                            print(f"Highlight failed (black vantage): {e}")
                            highlighted_black = frame_black

                        final_black_frame = highlighted_black

                    else:
                        # forced_vantage == "white"
                        # Always highlight from white vantage
                        rvec, tvec = white_transform['rvec'], white_transform['tvec']
                        inner_corners = white_transform['inner_corners']
                        board_size = white_transform['board_size']
                        square_size = white_transform['square_size']

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
                            print(f"Highlight failed (white vantage): {e}")
                            highlighted_white = frame_white

                        final_white_frame = highlighted_white

            # -------------------------------------------------
            # LAST STEP: Flip via mask if flip_mask_path is set
            # -------------------------------------------------
            if mask_float_3d is not None:
                final_black_frame = flip_frame_with_mask(final_black_frame, mask_float_3d)
                final_white_frame = flip_frame_with_mask(final_white_frame, mask_float_3d)

            # Convert to RGB & send to virtual cams
            out_black_rgb = cv2.cvtColor(final_black_frame, cv2.COLOR_BGR2RGB)
            out_white_rgb = cv2.cvtColor(final_white_frame, cv2.COLOR_BGR2RGB)

            v_cam4.send(out_black_rgb)   # LeRobot black
            v_cam6.send(out_black_rgb)   # OBS     black

            v_cam5.send(out_white_rgb)   # LeRobot white
            v_cam7.send(out_white_rgb)   # OBS     white

            # pace the loop once (last cam)
            v_cam7.sleep_until_next_frame()

            # Show windows
            cv2.imshow("Black side", final_black_frame)
            cv2.imshow("White side", final_white_frame)

            # Keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                if not board.is_game_over():
                    # BEFORE pushing the move, note who is about to move
                    last_move_was_white = board.turn  # True if White's turn
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
