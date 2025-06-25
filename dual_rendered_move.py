#!/usr/bin/env python3
"""
LeRobot move visualiser using a 3‑D rendered overlay. Similar to
``dual_highlight_move.py`` but the physical camera feeds are blended with a
rendered board and highlighted move squares.
"""

import cv2
import pyvirtualcam
import chess_vision
import time
import numpy as np
import subprocess
import chess
import chess.engine
import chess.svg
import cairosvg
import argparse
from pyvirtualcam import PixelFormat

from board_3d_overlay import (
    load_piece_models,
    render_board_state_with_move,
)

# --------------------------
# 0) 2-D Board Rendering (unchanged)
# --------------------------

def generate_board_image(board: chess.Board, last_move: chess.Move | None, size: int = 480) -> np.ndarray:
    svg_data = chess.svg.board(board=board, lastmove=last_move, size=size, coordinates=True)
    png_bytes = cairosvg.svg2png(bytestring=svg_data.encode("utf-8"), output_width=size, output_height=size)
    png_array = np.frombuffer(png_bytes, dtype=np.uint8)
    img_bgra = cv2.imdecode(png_array, cv2.IMREAD_UNCHANGED)
    img_bgr = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    x0 = (640 - size) // 2
    canvas[:, x0:x0 + size, :] = img_bgr
    return canvas


# --------------------------
# 1) Virtual Cameras
# --------------------------

def setup_virtual_cameras():
    print("Setting up 5 virtual cameras (4-8)…")
    subprocess.run(['sudo', 'modprobe', '-r', 'v4l2loopback'], check=True)
    time.sleep(1)
    subprocess.run([
        'sudo', 'modprobe', 'v4l2loopback',
        'devices=5',
        'video_nr=4,5,6,7,8',
        ('card_label="SO100 Black","SO100 White","OBS Black","OBS White","SO100 Board"'),
        'yuv420=1',
        'exclusive_caps=0'
    ], check=True)
    time.sleep(1)
    for dev in ('/dev/video4', '/dev/video5', '/dev/video6', '/dev/video7', '/dev/video8'):
        subprocess.run(['sudo', 'chmod', '666', dev], check=True)
    print("Virtual camera setup complete!")


# --------------------------
# 2) Stockfish helpers
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
# 3) Main
# --------------------------

def main():
    parser = argparse.ArgumentParser(description="3-D rendered move overlay")
    parser.add_argument(
        "--vantage",
        choices=["black", "white"],
        default=None,
        help="Force highlighting from one vantage (black/white).",
    )
    args = parser.parse_args()

    forced_vantage = args.vantage

    setup_virtual_cameras()

    board, engine = setup_chess_engine()
    if board is None:
        return

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

    print("Loading saved transforms…")
    chess_vision.load_saved_transform()
    black_transform = chess_vision.saved_transform.get(chess_vision.BLACK_SIDE_CAMERA)
    white_transform = chess_vision.saved_transform.get(chess_vision.WHITE_SIDE_CAMERA)
    if not black_transform or not white_transform:
        print("Missing transforms!")
        return

    models = load_piece_models("stl")

    with pyvirtualcam.Camera(640, 480, 30, fmt=PixelFormat.RGB, device='/dev/video4') as v_cam4, \
         pyvirtualcam.Camera(640, 480, 30, fmt=PixelFormat.RGB, device='/dev/video5') as v_cam5, \
         pyvirtualcam.Camera(640, 480, 30, fmt=PixelFormat.RGB, device='/dev/video6') as v_cam6, \
         pyvirtualcam.Camera(640, 480, 30, fmt=PixelFormat.RGB, device='/dev/video7') as v_cam7, \
         pyvirtualcam.Camera(640, 480, 30, fmt=PixelFormat.RGB, device='/dev/video8') as v_cam8:

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
        board_img_cache = generate_board_image(board, None)

        while True:
            ret_b, frame_black = cap_black.read()
            ret_w, frame_white = cap_white.read()
            if not (ret_b and ret_w):
                print("Camera read failure.")
                break

            highlight_for_black = False
            highlight_for_white = False
            if current_move:
                if forced_vantage is None:
                    highlight_for_black = last_move_was_white
                    highlight_for_white = not last_move_was_white
                elif forced_vantage == "black":
                    highlight_for_black = True
                else:
                    highlight_for_white = True

            final_black_frame = render_board_state_with_move(
                frame_black,
                board,
                models,
                (black_transform['rvec'], black_transform['tvec']),
                chess_vision.camera_matrix,
                current_move if highlight_for_black else None,
            )
            final_white_frame = render_board_state_with_move(
                frame_white,
                board,
                models,
                (white_transform['rvec'], white_transform['tvec']),
                chess_vision.camera_matrix,
                current_move if highlight_for_white else None,
            )


            v_cam4.send(cv2.cvtColor(final_black_frame, cv2.COLOR_BGR2RGB))
            v_cam6.send(cv2.cvtColor(final_black_frame, cv2.COLOR_BGR2RGB))
            v_cam5.send(cv2.cvtColor(final_white_frame, cv2.COLOR_BGR2RGB))
            v_cam7.send(cv2.cvtColor(final_white_frame, cv2.COLOR_BGR2RGB))
            v_cam8.send(cv2.cvtColor(board_img_cache, cv2.COLOR_BGR2RGB))
            v_cam8.sleep_until_next_frame()

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
                    board_img_cache = generate_board_image(board, current_move)
                else:
                    print(f"Game over. Result: {board.result()}")
            elif key == ord('r'):
                board.reset()
                current_move = None
                last_move_was_white = None
                board_img_cache = generate_board_image(board, None)
                print("Board reset.")

        cap_black.release()
        cap_white.release()
        cv2.destroyAllWindows()
        engine.quit()


if __name__ == "__main__":
    main()
