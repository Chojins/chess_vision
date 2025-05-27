#!/usr/bin/env python3
"""
LeRobot chess-move highlighter with an *extra* virtual camera that shows
a 2-D board diagram of the **desired** state + highlighted move squares.
"""

import cv2
import pyvirtualcam
import chess_vision
import time
import numpy as np
import subprocess
import chess
import chess.engine
import chess.svg               # NEW
import cairosvg                # NEW
import sys
import argparse
from pyvirtualcam import PixelFormat
import os

# --------------------------
# 0) 2-D Board Rendering
# --------------------------
def generate_board_image(board: chess.Board,
                         last_move: chess.Move | None,
                         size: int = 480) -> np.ndarray:
    """
    Returns a 640×480 RGB (uint8) image that contains a centred `size`×`size`
    board diagram (SVG->PNG->numpy). `last_move` is highlighted.
    """
    # 1. SVG string with python-chess
    svg_data = chess.svg.board(
        board=board,
        lastmove=last_move,
        size=size,
        coordinates=True
    )

    # 2. Rasterise to PNG bytes
    png_bytes = cairosvg.svg2png(bytestring=svg_data.encode("utf-8"),
                                 output_width=size,
                                 output_height=size)

    # 3. Convert bytes → ndarray (BGRA), then → BGR
    png_array = np.frombuffer(png_bytes, dtype=np.uint8)
    img_bgra = cv2.imdecode(png_array, cv2.IMREAD_UNCHANGED)
    img_bgr  = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)

    # 4. Embed in 640×480 canvas (black background)
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    x0 = (640 - size) // 2                   # centre horizontally
    canvas[:, x0:x0 + size, :] = img_bgr     # copy board into canvas

    return canvas


# --------------------------
# 1) Mask-based Flip Helpers
# --------------------------
def load_flip_mask(mask_path: str, width: int, height: int) -> np.ndarray:
    """
    Loads a black & white mask, converts it to 0-1 float, and
    ensures it matches the (width × height). Returns a 3-channel float mask.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Error: Could not read mask at {mask_path}")

    if (mask.shape[1] != width) or (mask.shape[0] != height):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    mask_float = mask.astype(np.float32) / 255.0
    return np.dstack([mask_float] * 3)

def flip_frame_with_mask(frame: np.ndarray, mask_float_3d: np.ndarray) -> np.ndarray:
    """Flip only the portion of `frame` where mask == 1."""
    flipped_frame = cv2.flip(frame, 1)
    return (flipped_frame * mask_float_3d +
            frame * (1.0 - mask_float_3d)).astype(np.uint8)


# --------------------------
# 2) Virtual Cameras
# --------------------------
def setup_virtual_cameras():
    """
    Creates FIVE virtual cameras:
        • 4 – SO100 Black   (LeRobot)
        • 5 – SO100 White   (LeRobot)
        • 6 – OBS Black     (OBS)
        • 7 – OBS White     (OBS)
        • 8 – SO100 Board   (2-D diagram)
    """
    print("Setting up 5 virtual cameras (4-8)…")
    subprocess.run(['sudo', 'modprobe', '-r', 'v4l2loopback'], check=True)
    time.sleep(1)

    subprocess.run([
        'sudo', 'modprobe', 'v4l2loopback',
        'devices=5',
        'video_nr=4,5,6,7,8',
        ('card_label="SO100 Black","SO100 White",'
         '"OBS Black","OBS White","SO100 Board"'),
        'yuv420=1',
        'exclusive_caps=0'
    ], check=True)
    time.sleep(1)

    for dev in ('/dev/video4', '/dev/video5',
                '/dev/video6', '/dev/video7',
                '/dev/video8'):
        subprocess.run(['sudo', 'chmod', '666', dev], check=True)

    print("Virtual camera setup complete!")


# --------------------------
# 3) Stockfish
# --------------------------
def setup_chess_engine():
    board = chess.Board()
    try:
        engine = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")
        print("Chess engine initialized successfully!")
        return board, engine
    except Exception as e:
        print(f"Error initializing chess engine: {e}")
        return None, None

def get_next_move(board, engine):
    return engine.play(board, chess.engine.Limit(time=0.1)).move


# --------------------------
# 4) Main
# --------------------------
def main():
    # ---- CLI ----
    parser = argparse.ArgumentParser(
        description="Highlight chess moves (black/white cams) plus 2-D board feed."
    )
    parser.add_argument("--vantage",
                        choices=["black", "white"],
                        default=None,
                        help="Force highlighting from one vantage (black/white).")
    parser.add_argument("--flip_mask",
                        type=str,
                        default=None,
                        help="Path to a black-white mask image for partial mirroring.")
    args = parser.parse_args()

    forced_vantage = args.vantage
    flip_mask_path = args.flip_mask

    # 1) Virtual cams
    setup_virtual_cameras()

    # 2) Engine
    board, engine = setup_chess_engine()
    if board is None:
        return

    # 3) Physical cams 0 & 2
    cap_black = cv2.VideoCapture(0)
    cap_white = cv2.VideoCapture(2)
    if not cap_black.isOpened() or not cap_white.isOpened():
        print("Could not open physical cameras!")
        engine.quit()
        return
    for cap in (cap_black, cap_white):
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 4) Load transforms
    print("Loading saved transforms…")
    chess_vision.load_saved_transform()
    black_transform = chess_vision.saved_transform.get(chess_vision.BLACK_SIDE_CAMERA)
    white_transform = chess_vision.saved_transform.get(chess_vision.WHITE_SIDE_CAMERA)
    if not black_transform or not white_transform:
        print("Missing transforms!")
        return

    # 5) Flip mask (optional)
    mask_float_3d = None
    if flip_mask_path and os.path.isfile(flip_mask_path):
        mask_float_3d = load_flip_mask(flip_mask_path, 640, 480)
        print(f"Loaded flip mask from {flip_mask_path}")

    # 6) Five virtual cams
    with pyvirtualcam.Camera(640, 480, 30, fmt=PixelFormat.RGB,
                             device='/dev/video4') as v_cam4, \
         pyvirtualcam.Camera(640, 480, 30, fmt=PixelFormat.RGB,
                             device='/dev/video5') as v_cam5, \
         pyvirtualcam.Camera(640, 480, 30, fmt=PixelFormat.RGB,
                             device='/dev/video6') as v_cam6, \
         pyvirtualcam.Camera(640, 480, 30, fmt=PixelFormat.RGB,
                             device='/dev/video7') as v_cam7, \
         pyvirtualcam.Camera(640, 480, 30, fmt=PixelFormat.RGB,
                             device='/dev/video8') as v_cam8:

        print(f'LeRobot cams  : {v_cam4.device}, {v_cam5.device}')
        print(f'OBS cams      : {v_cam6.device}, {v_cam7.device}')
        print(f'Board diagram : {v_cam8.device}')
        if forced_vantage:
            print(f"MODE: Fixed vantage → {forced_vantage}")
        else:
            print("MODE: Normal swapping (white move ⇒ black cam).")

        print("Keys: SPACE-next | r-reset | q-quit")

        current_move = None
        last_move_was_white = None
        board_img_cache = generate_board_image(board, None)  # start position

        while True:
            ret_b, frame_black = cap_black.read()
            ret_w, frame_white = cap_white.read()
            if not (ret_b and ret_w):
                print("Camera read failure.")
                break

            # ---------- Highlight physical frames ----------
            final_black_frame = frame_black
            final_white_frame = frame_white

            if current_move:
                move_str = current_move.uci()
                if forced_vantage is None:
                    # Normal (swap) mode
                    if last_move_was_white:
                        # White just moved → show on black cam
                        final_black_frame = chess_vision.highlight_chess_move(
                            frame_black, move_str,
                            black_transform['inner_corners'],
                            black_transform['board_size'],
                            black_transform['square_size'],
                            (black_transform['rvec'], black_transform['tvec']),
                            show_axes=False)
                    else:
                        final_white_frame = chess_vision.highlight_chess_move(
                            frame_white, move_str,
                            white_transform['inner_corners'],
                            white_transform['board_size'],
                            white_transform['square_size'],
                            (white_transform['rvec'], white_transform['tvec']),
                            show_axes=False)
                else:
                    # Fixed vantage
                    if forced_vantage == "black":
                        final_black_frame = chess_vision.highlight_chess_move(
                            frame_black, move_str,
                            black_transform['inner_corners'],
                            black_transform['board_size'],
                            black_transform['square_size'],
                            (black_transform['rvec'], black_transform['tvec']),
                            show_axes=False)
                    else:
                        final_white_frame = chess_vision.highlight_chess_move(
                            frame_white, move_str,
                            white_transform['inner_corners'],
                            white_transform['board_size'],
                            white_transform['square_size'],
                            (white_transform['rvec'], white_transform['tvec']),
                            show_axes=False)

            # Optional mask flip
            if mask_float_3d is not None:
                final_black_frame = flip_frame_with_mask(final_black_frame,
                                                         mask_float_3d)
                final_white_frame = flip_frame_with_mask(final_white_frame,
                                                         mask_float_3d)

            # ---------- Send to virtual cams ----------
            v_cam4.send(cv2.cvtColor(final_black_frame, cv2.COLOR_BGR2RGB))
            v_cam6.send(cv2.cvtColor(final_black_frame, cv2.COLOR_BGR2RGB))
            v_cam5.send(cv2.cvtColor(final_white_frame, cv2.COLOR_BGR2RGB))
            v_cam7.send(cv2.cvtColor(final_white_frame, cv2.COLOR_BGR2RGB))
            v_cam8.send(cv2.cvtColor(board_img_cache, cv2.COLOR_BGR2RGB))

            v_cam8.sleep_until_next_frame()

            # ---------- GUI preview ----------
            cv2.imshow("Black cam", final_black_frame)
            cv2.imshow("White cam", final_white_frame)
            cv2.imshow("Board diagram", board_img_cache)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                if not board.is_game_over():
                    last_move_was_white = board.turn
                    next_move = get_next_move(board, engine)
                    print(f"Engine move: {next_move.uci()}")
                    board.push(next_move)
                    current_move = next_move
                    # Regenerate board diagram
                    board_img_cache = generate_board_image(board, current_move)
                else:
                    print(f"Game over. Result: {board.result()}")
            elif key == ord('r'):
                board.reset()
                current_move = None
                last_move_was_white = None
                board_img_cache = generate_board_image(board, None)
                print("Board reset.")

        # Cleanup
        cap_black.release()
        cap_white.release()
        cv2.destroyAllWindows()
        engine.quit()

# ----------
if __name__ == "__main__":
    main()
