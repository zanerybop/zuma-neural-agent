from pathlib import Path
import json
import time
import numpy as np
import torch
from mss import MSS
from .model import TinyPolicy
from .vision import Frames, capture
from .windows import GameWindow, down


def play(args):
    torch.set_num_threads(4)
    model = TinyPolicy()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint.get('format') != 1:
        raise ValueError('Unsupported checkpoint format')
    model.load_state_dict(checkpoint['model']); model.eval()
    window = GameWindow.find(args.title)
    stack, active, previous_f8 = Frames(), False, False
    started, last_action = time.monotonic(), 0.
    log = Path(args.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"{'LIVE INPUT' if args.control else 'DRY RUN'}; F8 start/pause; F9 or Esc stop. Duration: {args.seconds}s.", flush=True)
    try:
        with MSS() as sct, log.open('a', encoding='utf-8') as stream, torch.inference_mode():
            while time.monotonic()-started < args.seconds:
                tick = time.monotonic()
                if down(0x78) or down(0x1B):
                    break
                f8 = down(0x77)
                if f8 and not previous_f8:
                    active = not active
                    stack = Frames()
                    print(f"Active: {active}", flush=True)
                previous_f8 = f8
                if not window.focused():
                    if active:
                        print('Paused: game lost focus. F8 to resume.', flush=True)
                    active, stack = False, Frames()
                if active:
                    image = capture(sct, window)
                    if np.std(image) < 2:
                        raise RuntimeError('Capture is blank. Try windowed mode.')
                    aim, logits = model(torch.from_numpy(stack.push(image)).unsqueeze(0))
                    probabilities = logits.softmax(1)[0]
                    action = int(probabilities.argmax())
                    if float(probabilities[action]) < args.confidence or tick-last_action < args.cooldown:
                        action = 0
                    xy = aim[0].tolist()
                    if args.control:
                        window.act(xy, action)
                    if action:
                        last_action = tick
                    stream.write(json.dumps(dict(time=tick-started, aim=xy, action=action,
                                                 probabilities=probabilities.tolist(), control=args.control))+'\n')
                    stream.flush()
                time.sleep(max(0, 1/args.hz-(time.monotonic()-tick)))
    except KeyboardInterrupt:
        pass
