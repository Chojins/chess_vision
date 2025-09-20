#!/usr/bin/env python3
"""
Open your stage, switch the main viewport between wrist cameras, and on demand:
capture the viewport → run chess_vision → preview overlay in a UI panel.

Run:
  cd ~/isaacsim/_build/linux-x86_64/release
  ./python.sh /home/jacob/Documents/chess_vision/wrist_capture_and_overlay_min.py \
    --usd "/home/jacob/Documents/chess_sim/Sim_Stage.usd" \
    --blue "/World/so100_blue/gripper/Camera" \
    --red  "/World/so100_red/gripper/Camera" \
    --res 800 800 \
    --move e2e4
"""

import argparse, sys, traceback
import asyncio, time
from pathlib import Path  # you already have this

import numpy as np
import cv2

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})  # GUI

import omni.usd, omni.ui as ui, omni.kit.app as kit_app
from pxr import Sdf
from omni.kit.viewport.utility import get_active_viewport

from omni.kit.viewport.utility import capture_viewport_to_file  # if not already imported


# Try to import viewport capture helpers (not all builds expose both)
try:
    from omni.kit.viewport.utility import capture_viewport_to_buffer
except Exception:
    capture_viewport_to_buffer = None
try:
    from omni.kit.viewport.utility import capture_viewport_to_file
except Exception:
    capture_viewport_to_file = None

CAP_DIR = Path.home() / "Documents/chess_vision" / "captures"
CAP_DIR.mkdir(parents=True, exist_ok=True)


# Pull in your chess_vision code (uses your calibration + helpers)
CV_DIR = str(Path(__file__).resolve().parent)
if CV_DIR not in sys.path:
    sys.path.append(CV_DIR)
import chess_vision as cvmod


def rgba_any_to_bgr_uint8(img):
    """
    Accepts:
      - np.uint8 RGBA (H,W,4) or RGB (H,W,3)
      - float RGBA/RGB in [0,1]
    Returns:
      - np.uint8 BGR (H,W,3) suitable for OpenCV.
    """
    if img is None:
        return None
    arr = np.array(img, copy=False)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if arr.ndim != 3:
        return None
    if arr.shape[-1] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    if arr.shape[-1] == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return None


def bgr_to_rgba_uint8(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd", required=True)
    ap.add_argument("--blue", default="/World/so100_blue/gripper/Camera")
    ap.add_argument("--red",  default="/World/so100_red/gripper/Camera")
    ap.add_argument("--res",  nargs=2, type=int, default=[800, 800])
    ap.add_argument("--move", default="e2e4", help="UCI move to highlight")
    args = ap.parse_args()

    # Open stage + let it settle a few frames
    usd_ctx = omni.usd.get_context()
    usd_ctx.open_stage(args.usd)
    for _ in range(30):
        simulation_app.update()

    stage = usd_ctx.get_stage()
    if stage is None:
        print("[ERROR] Stage failed to open.")
        simulation_app.close()
        sys.exit(1)

    vp = get_active_viewport()
    if vp is None:
        print("[ERROR] No active viewport.")
        simulation_app.close()
        sys.exit(1)

    # Set resolution if supported
    try:
        vp.set_texture_resolution(args.res[0], args.res[1])
    except Exception:
        pass

    # Start on BLUE camera
    vp.camera_path = Sdf.Path(args.blue)

    # UI: docked control panel + overlay preview
    window = ui.Window(
        "Wrist Cameras",
        width=args.res[0] + 240,
        height=args.res[1] + 220,
        dockPreference=ui.DockPreference.LEFT,
    )
    window.visible = True

    overlay_provider = ui.ByteImageProvider()  # for the processed preview
    status = {"text": "Ready.", "last_err": ""}

    def set_status(msg):
        status["text"] = msg
        status_label.text = msg
        print(f"[STATUS] {msg}", flush=True)

    # capture helper that tries buffer first, then file as fallback
    TEMP_PNG = str(Path.home() / ".isaac_tmp_viewport.png")

    def capture_and_detect():
        if capture_viewport_to_file is None:
            set_status("Viewport file-capture API not available in this build.")
            return

        try:
            set_status("Capturing viewport → file…")
            helper = capture_viewport_to_file(vp, TEMP_PNG, is_hdr=False)  # returns awaitable

            def _cam_name():
                return "blue" if str(vp.camera_path) == args.blue else "red"

            async def _wait_then_process():
                try:
                    await helper.wait_for_result()  # wait until PNG is fully written

                    # Load raw capture (BGR)
                    bgr = cv2.imread(TEMP_PNG, cv2.IMREAD_COLOR)
                    if bgr is None:
                        set_status("Capture produced no image (file read failed).")
                        return

                    # Save RAW frame
                    ts = time.strftime("%Y%m%d-%H%M%S")
                    cam = _cam_name()
                    raw_path = CAP_DIR / f"raw_{cam}_{ts}.png"
                    cv2.imwrite(str(raw_path), bgr)

                    # Run detection + overlay
                    use_white_side = (cam == "blue")
                    try:
                        ic, board_size, sq, pose = cvmod.find_chessboard_corners(bgr, use_white_side)
                        result = cvmod.highlight_chess_move(bgr, args.move, ic, board_size, sq, pose, undistort=False)

                        # Save OVERLAY frame
                        overlay_path = CAP_DIR / f"overlay_{cam}_{ts}.png"
                        cv2.imwrite(str(overlay_path), result)

                        # Update UI preview
                        rgba_out = cv2.cvtColor(result, cv2.COLOR_BGR2RGBA)
                        overlay_provider.set_data_array(rgba_out, [rgba_out.shape[1], rgba_out.shape[0]])

                        set_status(f"Saved: {raw_path.name}, {overlay_path.name}")
                    except Exception as e:
                        # Detection failed: still show raw in preview and report
                        rgba_raw = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
                        overlay_provider.set_data_array(rgba_raw, [rgba_raw.shape[1], rgba_raw.shape[0]])
                        set_status(f"Saved RAW ({raw_path.name}). Detection failed: {type(e).__name__}: {e}")

                except Exception as e:
                    set_status(f"Capture failed: {type(e).__name__}: {e}")

            asyncio.ensure_future(_wait_then_process())
        except Exception as e:
            set_status(f"Capture setup error: {type(e).__name__}: {e}")




    with window.frame:
        with ui.VStack(spacing=8, height=0):
            ui.Label("Viewport Wrist Camera Switcher", style={"font_size": 18})
            with ui.HStack(height=0, spacing=8):
                def to_blue():
                    vp.camera_path = Sdf.Path(args.blue)
                    set_status("Switched to BLUE camera.")
                def to_red():
                    vp.camera_path = Sdf.Path(args.red)
                    set_status("Switched to RED camera.")
                ui.Button("Blue view", clicked_fn=to_blue)
                ui.Button("Red view",  clicked_fn=to_red)
            ui.Separator()
            with ui.HStack(height=0, spacing=8):
                ui.Button("Capture & Detect", clicked_fn=capture_and_detect)
                status_label = ui.Label(status["text"])
            ui.Label("Preview (processed):")
            ui.ImageWithProvider(overlay_provider, width=args.res[0], height=args.res[1])

    app = kit_app.get_app()
    while app.is_running():
        simulation_app.update()

    simulation_app.close()


if __name__ == "__main__":
    main()
