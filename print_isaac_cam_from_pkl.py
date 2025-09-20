#!/usr/bin/env python3
import argparse, pickle, sys, math, re
from typing import Any, Dict, Iterable, Tuple, Optional

try:
    import numpy as np
except Exception as e:
    print("This script needs numpy. Please install it (pip install numpy).")
    raise

def norm_key(k: str) -> str:
    # Lowercase, strip non-alnum to match variants like image_size / imageSize
    return re.sub(r'[^a-z0-9]', '', k.lower())

def iter_items(obj: Any, path: str=""):
    """Yield (path, key, value) for dict-like contents recursively."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else k
            yield from iter_items(v, p)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            p = f"{path}[{i}]"
            yield from iter_items(v, p)
    else:
        # leaf
        yield (path, None, obj)

def find_first_matrix(dic: Any) -> Optional[np.ndarray]:
    """Try to find camera matrix K in various common keys."""
    candidates = [
        "camera_matrix", "cameramatrix", "k", "mtx", "intrinsic", "intrinsics",
        "intrinsic_matrix", "kmatrix", "kalibrationmatrix", "calibmatrix"
    ]
    if isinstance(dic, dict):
        # quick direct lookups
        for k in list(dic.keys()):
            nk = norm_key(k)
            if nk in candidates:
                try:
                    arr = np.array(dic[k], dtype=float).reshape(3,3)
                    return arr
                except Exception:
                    pass
    # recursive search
    for p, _, v in iter_items(dic):
        if isinstance(v, (list, tuple, np.ndarray)):
            arr = np.array(v, dtype=float)
            if arr.size == 9:
                arr = arr.reshape(3,3)
                # Heuristic: bottom row ~ [0,0,1]
                if np.allclose(arr[2], [0,0,1], atol=1e-3):
                    return arr
    return None

def find_dist_coeffs(dic: Any) -> Optional[np.ndarray]:
    keys = ["dist_coeffs","distortion_coefficients","distortion","dist","d","distcoeffs"]
    # direct
    if isinstance(dic, dict):
        for k in dic.keys():
            if norm_key(k) in keys:
                try:
                    return np.array(dic[k], dtype=float).flatten()
                except Exception:
                    pass
    # recursive
    for p, _, v in iter_items(dic):
        if isinstance(v, (list, tuple, np.ndarray)) and 4 <= np.size(v) <= 14:
            arr = np.array(v, dtype=float).flatten()
            # heuristic: small-ish coeffs
            if np.all(np.abs(arr) < 5):
                return arr
    return None

def find_resolution(dic: Any) -> Optional[Tuple[int,int]]:
    # Common patterns: image_size, img_size, frame_size, resolution, image_shape, (width,height), (cols,rows)
    # 1) direct tuple-like fields
    for k in getattr(dic, "keys", lambda: [])():
        nk = norm_key(k)
        if nk in ["imagesize","imgsize","framesize","resolution","size"]:
            try:
                v = dic[k]
                a = np.array(v).flatten().astype(int)
                if a.size >= 2:
                    W, H = int(a[0]), int(a[1])
                    if W > 0 and H > 0:
                        return (W, H)
            except Exception:
                pass

    # 2) width/height or cols/rows style
    width = dic.get("image_width") or dic.get("width") or dic.get("W") or dic.get("cols")
    height = dic.get("image_height") or dic.get("height") or dic.get("H") or dic.get("rows")
    if width and height:
        try:
            return (int(width), int(height))
        except Exception:
            pass

    # 3) image_shape (H,W[,C])
    ishape = dic.get("image_shape") or dic.get("img_shape") or dic.get("shape")
    if ishape is not None:
        try:
            a = np.array(ishape).flatten().astype(int)
            if a.size >= 2:
                H, W = int(a[0]), int(a[1])
                if W > 0 and H > 0:
                    return (W, H)
        except Exception:
            pass

    # 4) recursive search for any of the above in nested dicts
    for p, _, v in iter_items(dic):
        if isinstance(v, dict):
            r = find_resolution(v)
            if r:
                return r
        elif isinstance(v, (list, tuple)) and len(v) >= 2:
            a = np.array(v).flatten()
            if a.size >= 2 and np.issubdtype(a.dtype, np.integer):
                W, H = int(a[0]), int(a[1])
                if 32 <= W <= 20000 and 32 <= H <= 20000:  # sanity bounds
                    return (W, H)
    return None

def pretty_top_level(dic: Dict[str, Any]):
    print("Top-level keys in pickle:")
    for k in dic.keys():
        v = dic[k]
        t = type(v).__name__
        if isinstance(v, (list, tuple, np.ndarray)):
            try:
                shape = np.array(v).shape
                print(f"  - {k}  ({t}, shape={shape})")
            except Exception:
                print(f"  - {k}  ({t})")
        elif isinstance(v, dict):
            print(f"  - {k}  (dict, {len(v)} keys)")
        else:
            print(f"  - {k}  ({t})")

def main():
    ap = argparse.ArgumentParser(description="Print Isaac Sim camera fields from an OpenCV calibration .pkl")
    ap.add_argument("pkl", help="Path to calibration pickle")
    ap.add_argument("--resolution", nargs=2, type=int, metavar=("W","H"),
                    help="Override image resolution if not present in the pickle")
    ap.add_argument("--sensor-mm", nargs=2, type=float, metavar=("SW_MM","SH_MM"),
                    help="Real sensor width/height in millimeters (optional)")
    args = ap.parse_args()

    with open(args.pkl, "rb") as f:
        calib = pickle.load(f)

    if not isinstance(calib, dict):
        print("Note: pickle root is not a dict; attempting to coerce common structures.")
        # Some dumps store a tuple like (K, dist, rvecs, tvecs, ...). Wrap it.
        calib = {"root": calib}

    pretty_top_level(calib)

    K = find_first_matrix(calib)
    if K is None:
        sys.exit("\nERROR: Could not find a 3x3 camera matrix (K) in the pickle.")
    fx, fy, cx, cy, skew = float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2]), float(K[0,1])

    if args.resolution:
        W, H = args.resolution
        res_source = "CLI --resolution"
    else:
        res = find_resolution(calib)
        if res is None:
            sys.exit("\nERROR: Could not determine image resolution from the pickle.\n"
                     "Fix: rerun with --resolution W H (e.g., --resolution 1920 1080).")
        W, H = res
        res_source = "pickle"

    dist = find_dist_coeffs(calib)
    # Optional extrinsics
    rvec = calib.get("rvec") or calib.get("rvecs")
    tvec = calib.get("tvec") or calib.get("tvecs")

    def fmt(x): return f"{x:.6f}"

    print("\n== Intrinsics ==")
    print(f"Source of resolution: {res_source}")
    print(f"Resolution (px): {W} x {H}")
    print(f"fx={fmt(fx)}, fy={fmt(fy)}, cx={fmt(cx)}, cy={fmt(cy)}, skew={fmt(skew)}")

    # Mapping A: pixel-as-mm
    sw, sh = float(W), float(H)
    f_mm_A = fx * sw / W  # == fx
    hA_A, vA_A = sw, sh
    hOff_A = (cx - W/2.0) * (sw / W)
    vOff_A = (H/2.0 - cy) * (sh / H)  # y sign flip

    print("\n== Isaac Camera fields (paste into Camera prim) ==")
    print("projection: perspective")
    print(f"focalLength (mm):            {fmt(f_mm_A)}")
    print(f"horizontalAperture (mm):     {fmt(hA_A)}")
    print(f"verticalAperture (mm):       {fmt(vA_A)}")
    print(f"horizontalApertureOffset:    {fmt(hOff_A)}")
    print(f"verticalApertureOffset:      {fmt(vOff_A)}")
    print("Set Render Product resolution:", f"{W} x {H}")

    if args.sensor_mm:
        sw_mm, sh_mm = args.sensor_mm
        f_mm_B = fx * sw_mm / W
        vA_B = f_mm_B * H / fy
        hA_B = sw_mm
        hOff_B = (cx - W/2.0) * (hA_B / W)
        vOff_B = (H/2.0 - cy) * (vA_B / H)
        print("\n-- Alternative using real sensor size --")
        print(f"focalLength (mm):            {fmt(f_mm_B)}")
        print(f"horizontalAperture (mm):     {fmt(hA_B)}   # sensor width")
        print(f"verticalAperture (mm):       {fmt(vA_B)}   # derived to match fy")
        print(f"horizontalApertureOffset:    {fmt(hOff_B)}")
        print(f"verticalApertureOffset:      {fmt(vOff_B)}")

    print("\n== Distortion ==")
    if dist is not None and dist.size > 0:
        labels = ["k1","k2","p1","p2","k3","k4","k5","k6","s1","s2","s3","s4","tx","ty"]
        named = ", ".join(f"{labels[i]}={fmt(dist[i])}" for i in range(min(len(dist), len(labels))))
        print("Assuming OpenCV pinhole model. Coeffs:", named)
        print("Note: enable camera post-process lens distortion in Isaac if you want distorted renders.")
    else:
        print("No distortion coefficients found.")

    if rvec is not None and tvec is not None:
        try:
            r = np.array(rvec, dtype=float).reshape(-1)[:3]
            t = np.array(tvec, dtype=float).reshape(-1)[:3]
            theta = np.linalg.norm(r)
            if theta < 1e-12:
                R = np.eye(3)
            else:
                k = r / theta
                Kx = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]], dtype=float)
                R = np.eye(3) + math.sin(theta)*Kx + (1-math.cos(theta))*(Kx@Kx)
            Rwc = R.T
            twc = (-Rwc @ t.reshape(3,1)).reshape(3)
            B = np.diag([1,-1,-1])    # OpenCV -> USD basis flip
            Rusd = B @ Rwc
            tusd = B @ twc
            # matrix -> quat
            t0 = np.trace(Rusd)
            if t0 > 0:
                S = math.sqrt(t0 + 1.0) * 2.0
                qw = 0.25 * S
                qx = (Rusd[2,1] - Rusd[1,2]) / S
                qy = (Rusd[0,2] - Rusd[2,0]) / S
                qz = (Rusd[1,0] - Rusd[0,1]) / S
            else:
                i = np.argmax(np.diag(Rusd))
                if i == 0:
                    S = math.sqrt(1.0 + Rusd[0,0] - Rusd[1,1] - Rusd[2,2]) * 2.0
                    qw = (Rusd[2,1] - Rusd[1,2]) / S
                    qx = 0.25 * S
                    qy = (Rusd[0,1] + Rusd[1,0]) / S
                    qz = (Rusd[0,2] + Rusd[2,0]) / S
                elif i == 1:
                    S = math.sqrt(1.0 + Rusd[1,1] - Rusd[0,0] - Rusd[2,2]) * 2.0
                    qw = (Rusd[0,2] - Rusd[2,0]) / S
                    qx = (Rusd[0,1] + Rusd[1,0]) / S
                    qy = 0.25 * S
                    qz = (Rusd[1,2] + Rusd[2,1]) / S
                else:
                    S = math.sqrt(1.0 + Rusd[2,2] - Rusd[0,0] - Rusd[1,1]) * 2.0
                    qw = (Rusd[1,0] - Rusd[0,1]) / S
                    qx = (Rusd[0,2] + Rusd[2,0]) / S
                    qy = (Rusd[1,2] + Rusd[2,1]) / S
                    qz = 0.25 * S
            print("\n== Extrinsics (optional) → paste into Camera Xform ==")
            print(f"Translate (m): ({tusd[0]:.6f}, {tusd[1]:.6f}, {tusd[2]:.6f})")
            print(f"Orient (quat x,y,z,w): ({qx:.6f}, {qy:.6f}, {qz:.6f}, {qw:.6f})")
        except Exception:
            print("\nExtrinsics present but could not be parsed; skipping (that’s fine if you only need intrinsics).")
    else:
        print("\nNo rvec/tvec found (pose is optional).")

if __name__ == "__main__":
    main()
