from datetime import datetime, timezone
from pathlib import Path
import json
import time
import numpy as np
from mss import MSS
from .vision import Frames, capture
from .windows import GameWindow, down


def record(args):
    window = GameWindow.find(args.title)
    folder = Path(args.output) / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder.mkdir(parents=True)
    (folder/'session.json').write_text(json.dumps(dict(title=window.title, hz=args.hz, format=1), indent=2))
    print(f"Session: {folder}\nF8: start/pause recording. F9 or Esc: save and stop. Only gameplay should be recorded.", flush=True)
    observations, targets, actions, timestamps = [], [], [], []
    stack, active, previous, shard, total = Frames(), False, {}, 0, 0
    started = time.monotonic()

    def flush():
        nonlocal shard, total
        if not actions:
            return
        np.savez_compressed(folder/f'{shard:05d}.npz', observations=np.stack(observations),
                            targets=np.asarray(targets, dtype=np.float32), actions=np.asarray(actions, dtype=np.int64),
                            timestamps=np.asarray(timestamps))
        total += len(actions)
        print(f"Saved {total} samples", flush=True)
        observations.clear(); targets.clear(); actions.clear(); timestamps.clear()
        shard += 1

    try:
        with MSS() as sct:
            while time.monotonic()-started < args.seconds:
                tick = time.monotonic()
                keys = {k: down(k) for k in (0x77, 0x78, 0x1B, 1, 2)}
                edge = {k: value and not previous.get(k, False) for k, value in keys.items()}
                previous = keys
                if keys[0x78] or keys[0x1B]:
                    break
                if edge[0x77]:
                    active = not active
                    stack = Frames()
                    print(f"Recording: {active}", flush=True)
                if active and window.focused():
                    # Observe before waiting for the action, so labels describe the next decision.
                    observation = stack.push(capture(sct, window))
                    deadline = tick + 1/args.hz
                    action, xy = 0, window.cursor()
                    while time.monotonic() < deadline and window.focused():
                        if down(0x78) or down(0x1B) or down(0x77):
                            break
                        for key, label in ((1, 1), (2, 2)):
                            state = down(key)
                            if state and not previous[key] and action == 0:
                                action, xy = label, window.cursor()
                            previous[key] = state
                        time.sleep(0.002)
                    if window.focused() and all(0 <= v <= 1 for v in xy):
                        observations.append(observation); targets.append(xy); actions.append(action)
                        timestamps.append(tick-started)
                        if len(actions) >= 256:
                            flush()
                else:
                    stack = Frames()
                time.sleep(max(0, 1/args.hz-(time.monotonic()-tick)))
    except KeyboardInterrupt:
        pass
    finally:
        flush()
    print(f"Finished: {total} samples in {folder}", flush=True)
