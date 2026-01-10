import os, chess, chess.engine
engine_path = os.environ.get("STOCKFISH_PATH", "stockfish")
try:
    eng = chess.engine.SimpleEngine.popen_uci(engine_path)
    print("Engine OK at:", engine_path)
    print("ID:", eng.id)
    info = eng.analyse(chess.Board(), chess.engine.Limit(depth=8))
    pv = info.get('pv', [])
    print("PV length:", len(pv))
finally:
    try: eng.quit()
    except: pass