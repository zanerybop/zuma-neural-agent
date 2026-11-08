"""Index actual victory evidence across recorded attempts; never infer unseen wins."""
import json
from pathlib import Path
from .board import BoardVision


def index(root='runs', game=None):
    root = Path(root)
    vision = BoardVision(game) if game is not None else BoardVision()
    expected = [(stage, level, board) for stage in range(1, 14)
                for level, board in enumerate(vision.stages[f'stage{stage}'].split(','), 1)]
    wins = {}
    for path in sorted(root.glob('campaign-*/victories.json')):
        for row in json.loads(path.read_text(encoding='utf-8')):
            evidence = Path(row['evidence'])
            if evidence.exists():
                wins.setdefault((row['stage'], row['level']), dict(**row, attempt=str(path.parent)))
    manifest = dict(total_levels=len(expected), confirmed_levels=len(wins),
                    all_levels_confirmed=all((s, l) in wins for s, l, _ in expected),
                    levels=[dict(stage=s, level=l, board=b, status='won' if (s, l) in wins else 'unconfirmed',
                                 evidence=wins.get((s, l))) for s, l, b in expected])
    (root/'campaign-progress.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(dict(total=manifest['total_levels'], won=manifest['confirmed_levels'],
                          all_levels_confirmed=manifest['all_levels_confirmed'],
                          latest=[list(k) for k in sorted(wins)[-5:]])))
    return manifest


if __name__ == '__main__':
    index()
