#!/usr/bin/env python3
"""
LeRobot move visualiser with 3‑D rendered overlay that ALWAYS feeds the virtual
cameras at 30 fps, even while overlay rendering is busy.

Author: Jacob Gertsch & ChatGPT (2025‑07‑23)
"""

import cv2, pyvirtualcam, threading, time, queue, subprocess, argparse
import numpy as np, chess, chess.engine, chess.svg, cairosvg, chess_vision
from pyvirtualcam import PixelFormat
from board_3d_overlay import load_piece_models, generate_board_overlay_with_move, composite_overlay

# ---------------------------------------------------------------------------
# Global constants
WIDTH, HEIGHT, FPS = 640, 480, 30

# ---------------------------------------------------------------------------
# Light‑weight helpers
def generate_board_image(board, last_move, size=480):
    svg = chess.svg.board(board=board, lastmove=last_move, size=size, coordinates=True)
    png = cairosvg.svg2png(bytestring=svg.encode(), output_width=size, output_height=size)
    img = cv2.cvtColor(cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED),
                       cv2.COLOR_BGRA2BGR)
    canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    x0 = (WIDTH - size) // 2
    canvas[:, x0:x0+size] = img
    return canvas

def safe_read(cap, fallback):
    ok, frame = cap.read()
    return frame if ok else fallback

# ---------------------------------------------------------------------------
# Virtual cameras (reload kernel module every run)
def setup_virtual_cams():
    subprocess.run(['sudo','modprobe','-r','v4l2loopback'], check=True)
    time.sleep(1)
    subprocess.run([
        'sudo','modprobe','v4l2loopback',
        'devices=5','video_nr=4,5,6,7,8',
        ('card_label="SO100 Black","SO100 White","OBS Black","OBS White","SO100 Board"'),
        'yuv420=1','exclusive_caps=0'
    ], check=True)
    for dev in ('/dev/video4','/dev/video5','/dev/video6','/dev/video7','/dev/video8'):
        subprocess.run(['sudo','chmod','666',dev], check=True)

# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vantage", choices=["black","white"])
    args = ap.parse_args()
    fixed_vantage = args.vantage

    setup_virtual_cams()
    board = chess.Board()
    engine = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")

    # physical cams (0 = black side, 2 = white side)
    cap_b, cap_w = cv2.VideoCapture(0), cv2.VideoCapture(2)
    for c in (cap_b, cap_w):
        c.set(cv2.CAP_PROP_FPS, FPS)
        c.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    # camera calibration
    chess_vision.load_saved_transform()
    bt, wt = (chess_vision.saved_transform[k] for k in
              (chess_vision.BLACK_SIDE_CAMERA, chess_vision.WHITE_SIDE_CAMERA))
    models = load_piece_models("stl")

    # shared (thread‑safe) buffers
    last_black = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    last_white = last_black.copy()
    last_board = generate_board_image(board, None)
    ov_b = ov_w = None          # current overlays
    move_q = queue.SimpleQueue() # board objects to render

    # -----------------------------------------------------------------------
    # Worker thread: heavy overlay rendering
    def renderer():
        nonlocal ov_b, ov_w, last_board
        while True:
            b_state, mv = move_q.get()            # blocks
            last_board = generate_board_image(b_state, mv)
            ov_b = generate_board_overlay_with_move(
                b_state, models, (bt['rvec'],bt['tvec']),
                chess_vision.camera_matrix, WIDTH, HEIGHT, mv)
            ov_w = generate_board_overlay_with_move(
                b_state, models, (wt['rvec'],wt['tvec']),
                chess_vision.camera_matrix, WIDTH, HEIGHT, mv)

    threading.Thread(target=renderer, daemon=True).start()

    # -----------------------------------------------------------------------
    # Virtual cameras (cam4 master‑clocks the loop)
    with pyvirtualcam.Camera(WIDTH,HEIGHT,FPS,fmt=PixelFormat.RGB,device='/dev/video4') as v4,\
         pyvirtualcam.Camera(WIDTH,HEIGHT,FPS,fmt=PixelFormat.RGB,device='/dev/video5') as v5,\
         pyvirtualcam.Camera(WIDTH,HEIGHT,FPS,fmt=PixelFormat.RGB,device='/dev/video6') as v6,\
         pyvirtualcam.Camera(WIDTH,HEIGHT,FPS,fmt=PixelFormat.RGB,device='/dev/video7') as v7,\
         pyvirtualcam.Camera(WIDTH,HEIGHT,FPS,fmt=PixelFormat.RGB,device='/dev/video8') as v8:

        print("SPACE = next move • r = reset • q = quit")
        cur_move, last_move_white = None, None

        while True:
            # 1 grab or reuse frames
            last_black = safe_read(cap_b, last_black)
            last_white = safe_read(cap_w, last_white)

            show_b, show_w = last_black, last_white
            if cur_move:
                hb =  last_move_white if fixed_vantage is None else fixed_vantage=="black"
                hw = not last_move_white if fixed_vantage is None else fixed_vantage=="white"
                if hb and ov_b is not None: show_b = composite_overlay(last_black , ov_b)
                if hw and ov_w is not None: show_w = composite_overlay(last_white, ov_w)

            # 2 send to all virtual cams (convert once to RGB)
            rgb_b, rgb_w = map(lambda x: cv2.cvtColor(x,cv2.COLOR_BGR2RGB),
                               (show_b, show_w))
            board_rgb = cv2.cvtColor(last_board, cv2.COLOR_BGR2RGB)
            v4.send(rgb_b); v6.send(rgb_b)    # black views
            v5.send(rgb_w); v7.send(rgb_w)    # white views
            v8.send(board_rgb)                # 2‑D diagram
            v4.sleep_until_next_frame()       # 30 fps metronome

            # 3 preview & key handling
            cv2.imshow("B", show_b); cv2.imshow("W", show_w); cv2.imshow("D", last_board)
            k = cv2.waitKey(1) & 0xFF
            if k==ord('q'): break
            if k==ord(' '):
                if board.is_game_over():
                    print("Game over:", board.result()); continue
                last_move_white = board.turn
                nxt = engine.play(board, chess.engine.Limit(time=0.1)).move
                print("Move:", nxt.uci()); board.push(nxt); cur_move = nxt
                move_q.put((board.copy(), nxt))    # hand off to renderer
            if k==ord('r'):
                board.reset(); cur_move=None; last_move_white=None
                last_board = generate_board_image(board, None)
                ov_b = ov_w = None
                with move_q.mutex: move_q.queue.clear()

    cap_b.release(); cap_w.release(); cv2.destroyAllWindows(); engine.quit()

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
