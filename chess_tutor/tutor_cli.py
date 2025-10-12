"""Dedicated CLI entry point to avoid name collisions."""
try:
    from .tutor_core import ChessTutor, build_index
    from .config import SHOW_GAME_BOARDS
except ImportError:
    from tutor_core import ChessTutor, build_index  # type: ignore
    from config import SHOW_GAME_BOARDS  # type: ignore

try:
    import chess
except ImportError:
    chess = None

COMMANDS = [
    ("explain <topic>", "Get an explanation about a chess topic"),
    ("position <FEN>", "Display a chess board from a FEN string"),
    ("<question> || fen:<FEN>", "Ask a question and analyze a position"),
    ("list games", "List all indexed games (numbered)"),
    ("games <prefix>", "Find games by move prefix (numbered)"),
    ("game <id|#n>", "Show full moves & final board of a game by id or number"),
    ("play <id|#n>", "Start interactive navigation of a game by id or number"),
    ("next / prev", "Step forward/backward in current game (after play)"),
    ("goto <ply>", "Jump to ply number in current game (0=start)"),
    ("progress", "Show your learning progress"),
    ("rebuild", "Rebuild the search index from PDFs"),
    ("reload_games", "Reload games from PDFs"),
    ("help", "Show this help menu"),
    ("quit", "Exit the tutor")
]


def print_help():
    print("\n♟️ Adaptive Chess Tutor ready.")
    print("Commands:")
    for cmd, desc in COMMANDS:
        print(f"  {cmd:<24} - {desc}")
    print()


def main():
    bot = ChessTutor()
    current_game_id = None
    current_ply = 0
    last_listed_games = []
    print_help()
    while True:
        q = input("You: ").strip()
        if q.lower() in ["quit", "exit"]:
            break
        elif q.lower() in ["help", "?", "h"]:
            print_help()
        elif q.lower() in ["rebuild", "recreate", "reindex"]:
            print("🔄 Rebuilding index from PDFs...")
            bot.index, bot.model, bot.docs, bot.metas = build_index()
            bot.progress = {}
            print("✅ Index rebuilt.")
        elif q.lower() in ["list games", "show games"]:
            last_listed_games = bot.list_games()
            if not last_listed_games:
                print("No games loaded.")
            else:
                for i, g in enumerate(last_listed_games, start=1):
                    print(f"[{i}] id=\"{g['id']}\" len={len(g['moves'])} final FEN={g['final_fen']}")
        elif q.startswith("explain "):
            topic = q.split(" ", 1)[1]
            print(bot.explain(topic))
        elif q.startswith("games "):
            query = q.split(" ", 1)[1]
            res = bot.find_games_by_move_prefix(query)
            last_listed_games = res
            if not res:
                print("No games found with that prefix.")
            else:
                for i, g in enumerate(res, start=1):
                    print(f"[{i}] id=\"{g['id']}\" moves:{' '.join(g['moves'][:20])} ... final FEN: {g['final_fen']}")
        elif q == "games":
            last_listed_games = bot.list_games()
            for i, g in enumerate(last_listed_games, start=1):
                print(f"[{i}] id=\"{g['id']}\" len={len(g['moves'])} final FEN={g['final_fen']}")
        elif q.startswith("position "):
            fen = q.split(" ", 1)[1]
            bot.set_position(fen)
        elif "|| fen:" in q:
            parts = q.split("|| fen:")
            question, fen = parts[0].strip(), parts[1].strip()
            print(bot.explain(question))
            print(bot.analyze_position(fen))
        elif q == "progress":
            bot.show_progress()
        elif q.startswith("game ") and not q.startswith("game_ply"):
            sel = q.split(" ", 1)[1].strip()
            target = sel.strip().strip('"').strip("'")
            game = None
            if target.startswith('#') or target.isdigit():
                try:
                    idx = int(target[1:] if target.startswith('#') else target) - 1
                    if 0 <= idx < len(last_listed_games):
                        game = last_listed_games[idx]
                except ValueError:
                    game = None
            if game is None:
                game = bot.get_game_by_id(target)
            if not game:
                print("Game not found. Use list games/games <prefix> to see numbered list.")
            else:
                print(f"Moves: {' '.join(game['moves'])}")
                if SHOW_GAME_BOARDS:
                    btxt = bot.render_board_ascii(game['moves'])
                    if btxt:
                        print(btxt)
        elif q.startswith("play "):
            sel = q.split(" ", 1)[1].strip()
            target = sel.strip().strip('"').strip("'")
            game = None
            if target.startswith('#') or target.isdigit():
                try:
                    idx = int(target[1:] if target.startswith('#') else target) - 1
                    if 0 <= idx < len(last_listed_games):
                        game = last_listed_games[idx]
                except ValueError:
                    game = None
            if game is None:
                game = bot.get_game_by_id(target)
            if not game:
                print("Game not found. Use list games/games <prefix> to see numbered list.")
            else:
                current_game_id = game['id']
                current_ply = 0
                board_txt, _, info = bot.board_after_ply(current_game_id, current_ply)
                print(info)
                if board_txt:
                    print(board_txt)
                print("Type next / prev / goto <ply> / help / quit to leave.")
        elif q.lower() == "next":
            if not current_game_id:
                print("No active game. Use play <id|#n> after listing games.")
            else:
                current_ply += 1
                board_txt, san_seq, info = bot.board_after_ply(current_game_id, current_ply)
                if board_txt is None:
                    print(info or "Cannot step.")
                else:
                    print(info)
                    print("SAN so far:", " ".join(san_seq))
                    print(board_txt)
        elif q.lower() == "prev":
            if not current_game_id:
                print("No active game. Use play <id|#n> after listing games.")
            else:
                current_ply -= 1
                if current_ply < 0:
                    current_ply = 0
                board_txt, san_seq, info = bot.board_after_ply(current_game_id, current_ply)
                if board_txt is None:
                    print(info or "Cannot step.")
                else:
                    print(info)
                    print("SAN so far:", " ".join(san_seq))
                    print(board_txt)
        elif q.startswith("goto "):
            if not current_game_id:
                print("No active game. Use play <id|#n> after listing games.")
            else:
                try:
                    ply = int(q.split(" ", 1)[1])
                except ValueError:
                    print("Provide a numeric ply.")
                    continue
                current_ply = ply
                board_txt, san_seq, info = bot.board_after_ply(current_game_id, current_ply)
                if board_txt is None:
                    print(info or "Cannot jump.")
                else:
                    print(info)
                    print("SAN so far:", " ".join(san_seq))
                    print(board_txt)
        elif q == "reload_games":
            bot.reload_games()
        else:
            print(bot.explain(q))


if __name__ == "__main__":
    main()
