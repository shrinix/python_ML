import chess
from backend.principles import attacked_pieces

def test_attack_persists():
    # Initial position: bishop attacks king
    board = chess.Board("rnbqk2r/pppp1ppp/4pn2/8/1bPP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 2 4")
    info = attacked_pieces.detect_info(board)
    arrows = attacked_pieces.visualize(board).get('arrows', [])
    assert any(a['from'] == 'b4' and a['to'] == 'e1' for a in arrows), "Arrow from b4 to e1 should be present"

    # Move that does NOT break the attack (white plays h2-h3)
    board.push(chess.Move.from_uci('h2h3'))
    info = attacked_pieces.detect_info(board)
    arrows = attacked_pieces.visualize(board).get('arrows', [])
    assert any(a['from'] == 'b4' and a['to'] == 'e1' for a in arrows), "Arrow from b4 to e1 should persist after d2d3"

    # Move that breaks the attack (bishop moves)
    board.push(chess.Move.from_uci('b4c3'))
    info = attacked_pieces.detect_info(board)
    arrows = attacked_pieces.visualize(board).get('arrows', [])
    assert not any(a['from'] == 'b4' and a['to'] == 'e1' for a in arrows), "Arrow from b4 to e1 should disappear after bishop moves"

if __name__ == "__main__":
    test_attack_persists()
    print("Regression test completed.")
