import cv2
import numpy as np
import pyrender
import trimesh
import chess

SQUARE_SIZE = 22.5  # millimeters
square_size_m = SQUARE_SIZE / 1000.0


def load_piece_models(models_dir):
    """Load STL models for each chess piece."""
    pieces = {}
    names = {
        'P': 'pawn',
        'N': 'knight',
        'B': 'bishop',
        'R': 'rook',
        'Q': 'queen',
        'K': 'king'
    }
    for color in ('white', 'black'):
        for symbol, name in names.items():
            key = (color[0] + symbol.upper())
            path = f"{models_dir}/{color}_{name}.stl"
            mesh = trimesh.load(path)
            pieces[key] = mesh
    return pieces


def square_to_board_coords(square):
    """Convert a chess square index (0..63) to board row/col."""
    row = 7 - chess.square_rank(square)
    col = chess.square_file(square)
    return row, col


def compute_piece_pose(row, col, height=0.0):
    T = np.eye(4, dtype=np.float32)
    T[0, 3] = col * square_size_m
    T[1, 3] = row * square_size_m
    T[2, 3] = height
    return T


def render_board_state(frame, board, models, pose, camera_matrix):
    rvec, tvec = pose
    R, _ = cv2.Rodrigues(rvec)
    T_board = np.eye(4)
    T_board[:3, :3] = R
    T_board[:3, 3] = tvec.squeeze()

    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.3, 0.3, 0.3])

    camera = pyrender.IntrinsicsCamera(
        fx=camera_matrix[0, 0],
        fy=camera_matrix[1, 1],
        cx=camera_matrix[0, 2],
        cy=camera_matrix[1, 2]
    )
    scene.add(camera, pose=T_board)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(light, pose=T_board)

    for square, piece in board.piece_map().items():
        row, col = square_to_board_coords(square)
        key = (('w' if piece.color == chess.WHITE else 'b') + piece.symbol().upper())
        if key not in models:
            continue
        mesh = pyrender.Mesh.from_trimesh(models[key], smooth=False)
        piece_pose = T_board @ compute_piece_pose(row, col)
        scene.add(mesh, pose=piece_pose)

    renderer = pyrender.OffscreenRenderer(frame.shape[1], frame.shape[0])
    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()

    overlay = color[:, :, :3]
    alpha = color[:, :, 3:] / 255.0
    blended = (overlay * alpha + frame * (1 - alpha)).astype(np.uint8)
    return blended

