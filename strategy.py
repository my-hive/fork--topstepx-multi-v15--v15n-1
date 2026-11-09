"""v15n-1 adopts v15n-2 MNQ pre-open continuation; causal close stop."""
import numpy as np

PARAMS = {"signal_time": 25200, "entry_time": 36000, "flat_time": 52200, "size": 3, "stop_points": 45}

def entry_risk(panel):
    return {symbol: np.full(len(df), PARAMS["stop_points"] * 2, dtype=float) for symbol, df in panel.items()}

def signals(panel):
    result = {}
    for symbol, df in panel.items():
        out = np.zeros(len(df), dtype=np.int64)
        if symbol != "MNQ":
            result[symbol] = out
            continue
        open_, close, sec = df.open.to_numpy(), df.close.to_numpy(), df.ct_sec.to_numpy()
        for indices in df.groupby("session", sort=False).indices.values():
            start = np.flatnonzero(sec[indices] == PARAMS["signal_time"])
            entry = np.flatnonzero(sec[indices] == PARAMS["entry_time"])
            if not len(start) or not len(entry):
                continue
            side = int(np.sign(close[indices[entry[0]]] - open_[indices[start[0]]]))
            entry_price = close[indices[entry[0]]]
            active = side != 0
            for j in range(entry[0], len(indices)):
                i = indices[j]
                if sec[i] >= PARAMS["flat_time"]:
                    active = False
                elif active and side * (close[i] - entry_price) <= -PARAMS["stop_points"]:
                    active = False
                if active:
                    out[i] = side * PARAMS["size"]
        result[symbol] = out
    return result
