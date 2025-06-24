import cv2
import chess
import numpy as np
import chess_vision


def main():
    # Load piece images (PNG files in images/pieces)
    pieces = chess_vision.load_piece_models("images/pieces", piece_size=80)
    if not pieces:
        print("No piece models found. Place PNG files in images/pieces.")
        return

    board = chess.Board()
    overlay = chess_vision.generate_board_overlay(board, pieces, square_size=80)

    # Use stored transform for camera 0 if available
    chess_vision.load_saved_transform()
    transform = chess_vision.saved_transform.get(chess_vision.BLACK_SIDE_CAMERA)
    if not transform:
        print("No transform data found in board_transform.json")
        return

    frame = cv2.imread("images/chess-board.jpg")
    if frame is None:
        print("Could not load sample image")
        return

    warped = chess_vision.render_board_overlay(frame, overlay, transform["inner_corners"])
    result = chess_vision.composite_overlay(frame, warped, alpha=0.8)

    cv2.imshow("3D Overlay Example", result)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
