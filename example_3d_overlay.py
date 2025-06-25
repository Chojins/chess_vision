import cv2
import chess
import chess_vision
from board_3d_overlay import load_piece_models, generate_board_overlay, composite_overlay

# Directory containing ``pawn.stl``, ``rook.stl`` etc.  Models are assumed to
# be exported in millimetres and are automatically scaled to metres on load.
MODELS_DIR = "stl"


def main():
    board = chess.Board()
    board.push(chess.Move.from_uci("e2e4"))
    models = load_piece_models(MODELS_DIR)

    cap = cv2.VideoCapture(chess_vision.WHITE_SIDE_CAMERA)
    if not cap.isOpened():
        print("Could not open camera")
        return

    chess_vision.load_saved_transform()

    current_transform = chess_vision.saved_transform.get(chess_vision.WHITE_SIDE_CAMERA)
    if not current_transform:
        print("Missing saved transform for white side camera")
        return

    inner = current_transform['inner_corners']
    board_size = current_transform['board_size']
    square_size = current_transform['square_size']
    pose = (current_transform['rvec'], current_transform['tvec'])

    # Grab a frame to determine the output size
    ret, frame = cap.read()
    if not ret:
        print("Could not read initial frame")
        return

    board_overlay = generate_board_overlay(
        board,
        models,
        pose,
        chess_vision.camera_matrix,
        frame.shape[1],
        frame.shape[0],
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        highlight = chess_vision.highlight_chess_move(
            frame,
            "e2e4",
            inner,
            board_size,
            square_size,
            pose,
            show_axes=False,
        )
        final = composite_overlay(highlight, board_overlay)
        cv2.imshow("3D Overlay", final)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


