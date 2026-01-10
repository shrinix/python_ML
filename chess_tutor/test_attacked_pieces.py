import chess
from backend.principles import attacked_pieces

def test_attacked_pieces_arrow():
    # FEN: rnbqk2r/pppp1ppp/4pn2/8/1bPP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 2 4
    board = chess.Board("rnbqk2r/pppp1ppp/4pn2/8/1bPP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 2 4")
    info = attacked_pieces.detect_info(board)
    print("Test output:", info)
    arrows = attacked_pieces.visualize(board).get('arrows', [])
    print("Arrows:", arrows)
    assert any(a['from'] == 'b4' and a['to'] == 'e1' for a in arrows), "Expected arrow from b4 to e1"

if __name__ == "__main__":
    test_attacked_pieces_arrow()
    print("Test completed.")
