import os
import cv2
import numpy as np
import pyrender
import trimesh
import chess

SQUARE_SIZE = 22.5  # millimeters
square_size_m = SQUARE_SIZE / 1000.0

# Pieces are rendered slightly transparent so the camera feed is still visible
PIECE_ALPHA = 0.7


def load_piece_models(models_dir, scale=0.001):
    """Load STL models for each chess piece.

    The directory is expected to contain files named ``pawn.stl`` ``rook.stl``
    and so on.  The same models are used for both colours. ``scale`` is applied
    to each mesh on load so models defined in millimetres can be converted to
    metres.
    """
    pieces = {}
    names = {
        'P': 'pawn',
        'N': 'knight',
        'B': 'bishop',
        'R': 'rook',
        'Q': 'queen',
        'K': 'king',
    }

    for symbol, name in names.items():
        path = os.path.join(models_dir, f"{name}.stl")
        mesh = trimesh.load(path)
        # Scale the mesh so coordinates in millimetres become metres
        if scale != 1.0:
            mesh.apply_scale(scale)
        pieces['w' + symbol] = mesh
        pieces['b' + symbol] = mesh

    return pieces


def square_to_board_coords(square):
    """Convert a chess square index (0..63) to board ``row`` and ``col``.

    ``python-chess`` numbers squares from ``a1`` (index ``0``) in the bottom left
    corner when viewed from White's side.  The board calibration used by the
    overlay routine follows the same convention, so no mirroring is required
    when mapping squares to board coordinates.
    """
    row = chess.square_rank(square)
    col = chess.square_file(square)
    return row, col


def compute_piece_pose(row, col, key, height=0.0):
    """Return a transformation that places a piece in the centre of ``row``, ``col``."""
    T = np.eye(4, dtype=np.float32)
    T[0, 3] = (col + 0.5) * square_size_m
    T[1, 3] = (row + 0.5) * square_size_m
    T[2, 3] = height

    #check if the piece is a white knight and rotate it 180 about z
    if key == 'wN':
        R = np.array([[np.cos(np.pi), -np.sin(np.pi), 0],
                       [np.sin(np.pi),  np.cos(np.pi), 0],
                       [0,              0,             1]])
        T[:3, :3] = R @ T[:3, :3]
    return T


def _camera_pose(pose):
    """Return a pose matrix for ``pyrender`` cameras from OpenCV rvec/tvec."""
    rvec, tvec = pose
    R, _ = cv2.Rodrigues(rvec)
    T_board = np.eye(4, dtype=np.float32)
    T_board[:3, :3] = R
    T_board[:3, 3] = tvec.squeeze()

    T_camera_in_board = np.linalg.inv(T_board)
    R_x180 = np.array([[1, 0, 0],
                       [0, -1, 0],
                       [0, 0, -1]], dtype=np.float32)
    T_xrotate = np.eye(4, dtype=np.float32)
    T_xrotate[:3, :3] = R_x180
    return T_camera_in_board @ T_xrotate


def render_board_state(frame, board, models, pose, camera_matrix):
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.3, 0.3, 0.3])

    camera = pyrender.IntrinsicsCamera(
        fx=camera_matrix[0, 0],
        fy=camera_matrix[1, 1],
        cx=camera_matrix[0, 2],
        cy=camera_matrix[1, 2],
        znear=0.01,
        zfar=10.0,
    )
    scene.add(camera, pose=_camera_pose(pose))

    light_pose = np.eye(4)
    light_pose[:3, 3] = [-40, 60, 80]
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
    scene.add(light, pose=light_pose)

    for square, piece in board.piece_map().items():
        row, col = square_to_board_coords(square)
        key = (('b' if piece.color == False else 'w') + piece.symbol().upper())
        if key not in models:
            continue

        WHITE_COLOR = [0.0, 0.0, 1.0, PIECE_ALPHA]  # Blue with alpha
        BLACK_COLOR = [1.0, 0.0, 0.0, PIECE_ALPHA]  # Red with alpha

        if piece.color == False:
            color = WHITE_COLOR
        else:
            color = BLACK_COLOR

        material = pyrender.MetallicRoughnessMaterial(baseColorFactor=color,
                                                      metallicFactor=0.0,
                                                      roughnessFactor=0.5)
        mesh = pyrender.Mesh.from_trimesh(models[key], material=material, smooth=False)
        scene.add(mesh, pose=compute_piece_pose(row, col, key))

    renderer = pyrender.OffscreenRenderer(frame.shape[1], frame.shape[0])
    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()

    overlay = color[:, :, :3]
    alpha = color[:, :, 3:] / 255.0
    blended = (overlay * alpha + frame * (1 - alpha)).astype(np.uint8)
    return blended


def generate_board_overlay(board, models, pose, camera_matrix, width, height):
    """Render the board once and return a transparent RGBA overlay."""
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.3, 0.3, 0.3])

    camera = pyrender.IntrinsicsCamera(
        fx=camera_matrix[0, 0],
        fy=camera_matrix[1, 1],
        cx=camera_matrix[0, 2],
        cy=camera_matrix[1, 2],
        znear=0.01,
        zfar=10.0,
    )
    scene.add(camera, pose=_camera_pose(pose))

    light_pose = np.eye(4)
    light_pose[:3, 3] = [-40, 60, 80]
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
    scene.add(light, pose=light_pose)

    for square, piece in board.piece_map().items():
        row, col = square_to_board_coords(square)
        key = ("b" if piece.color == False else "w") + piece.symbol().upper()
        mesh = models.get(key)
        if mesh is None:
            continue

        WHITE_COLOR = [0.0, 0.0, 1.0, PIECE_ALPHA]  # Blue with alpha
        BLACK_COLOR = [1.0, 0.0, 0.0, PIECE_ALPHA]  # Red with alpha

        if piece.color == False:
            color = WHITE_COLOR
        else:
            color = BLACK_COLOR

        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=color, metallicFactor=0.0, roughnessFactor=0.5
        )
        scene.add(
            pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False),
            pose=compute_piece_pose(row, col, key),
        )

    renderer = pyrender.OffscreenRenderer(width, height)
    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()
    return color


def composite_overlay(frame, overlay_rgba):
    """Blend a cached RGBA overlay with ``frame``."""
    overlay = overlay_rgba[:, :, :3]
    alpha = overlay_rgba[:, :, 3:] / 255.0
    return (overlay * alpha + frame * (1 - alpha)).astype(np.uint8)