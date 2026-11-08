import argparse
import json
from pathlib import Path


def positive(value):
    value = float(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('Must be positive')
    return value


def main():
    import sys
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description='Small neural agent for installed Zuma Deluxe')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('launch', help='Launch Steam Zuma Deluxe (3330)')
    sub.add_parser('windows', help='List visible window titles')
    sub.add_parser('doctor', help='Check Python, Torch, GPU and game window')
    snap = sub.add_parser('capture', help='Capture the game client area')
    snap.add_argument('--output', default='runs/capture.png')
    rec = sub.add_parser('record', help='Record human demonstrations')
    rec.add_argument('--output', default='data/demos')
    train = sub.add_parser('train', help='Train using human demonstrations')
    train.add_argument('--data', default='data/demos')
    train.add_argument('--output', default='runs/imitation')
    train.add_argument('--epochs', type=int, default=25)
    train.add_argument('--batch-size', type=int, default=64)
    train.add_argument('--lr', type=positive, default=0.0003)
    train.add_argument('--seed', type=int, default=42)
    train.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    play = sub.add_parser('play', help='Run a trained policy (dry run by default)')
    play.add_argument('--checkpoint', default='runs/imitation/best.pt')
    play.add_argument('--control', action='store_true', help='Enable mouse actions')
    play.add_argument('--confidence', type=float, default=.55)
    play.add_argument('--cooldown', type=positive, default=.25)
    play.add_argument('--log', default='runs/play.jsonl')
    for p in (snap, rec, play):
        p.add_argument('--title', default='Zuma Deluxe')
    for p in (rec, play):
        p.add_argument('--seconds', type=positive, default=600 if p is rec else 60)
        p.add_argument('--hz', type=positive, default=10)
    args = parser.parse_args()
    if args.command == 'train' and (args.epochs < 1 or args.batch_size < 1):
        parser.error('epochs and batch-size must be positive')
    if args.command == 'play' and not 0 <= args.confidence <= 1:
        parser.error('confidence must be between 0 and 1')
    try:
        if args.command == 'launch':
            import os
            os.startfile('steam://rungameid/3330')
        elif args.command == 'windows':
            from .windows import windows
            for hwnd, title in windows():
                print(f'{hwnd}: {title}')
        elif args.command == 'doctor':
            import sys
            import torch
            from .model import TinyPolicy
            from .windows import windows
            print(json.dumps(dict(python=sys.version, torch=torch.__version__, cuda=torch.cuda.is_available(),
                                  gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                                  parameters=sum(p.numel() for p in TinyPolicy().parameters()),
                                  game_windows=[t for _, t in windows() if 'zuma' in t.casefold()]), indent=2))
        elif args.command == 'capture':
            import numpy as np
            from mss import MSS
            from PIL import Image
            from .windows import GameWindow
            from .vision import capture
            window = GameWindow.find(args.title)
            if not window.focused():
                raise RuntimeError('Focus Zuma before capturing: screenshot capture reads visible pixels.')
            with MSS() as sct:
                rgb = capture(sct, window)
            if np.std(rgb) < 2:
                raise RuntimeError('Blank capture. Try windowed mode.')
            path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(path)
            print(f'{path}: {rgb.shape[1]}x{rgb.shape[0]}')
        elif args.command == 'record':
            from .record import record
            record(args)
        elif args.command == 'train':
            from .train import train
            train(args)
        elif args.command == 'play':
            from .play import play
            play(args)
    except (RuntimeError, ValueError, FileNotFoundError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
