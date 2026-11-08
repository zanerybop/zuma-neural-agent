"""A lightweight status report; does not move the mouse or interrupt recording."""
import json
from pathlib import Path
from .footage import metadata, alive


def last_row(path):
    if not path.exists():
        return None
    with path.open('rb') as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size-65536))
        lines = stream.read().splitlines()
    for line in reversed(lines):
        try:
            row=json.loads(line)
            if 'balls' in row:
                return row
        except (ValueError, UnicodeDecodeError):
            continue  # A writer may still be finishing the last row.
    return None


def status():
    root = Path(__file__).resolve().parent.parent
    progress_file = root/'runs/campaign-progress.json'
    progress = json.loads(progress_file.read_text(encoding='utf-8')) if progress_file.exists() else {}
    attempts = sorted([*(root/'runs').glob('campaign-*/attempt.json'),*(root/'runs').glob('gauntlet-*/attempt.json')],
                      key=lambda p:p.stat().st_mtime)
    folder = attempts[-1].parent if attempts else None
    result_path = folder/'result.json' if folder else None
    result = json.loads(result_path.read_text()) if result_path and result_path.exists() else None
    row = last_row(folder/'decisions.jsonl') if folder else None
    attempt=json.loads((folder/'attempt.json').read_text()) if folder else {}
    gauntlet_boards={json.loads(p.read_text())['board'] for p in (root/'runs').glob('gauntlet-*/sun-god.json')
                     if (p.parent/'sun-god-reached.png').exists()}
    active_recordings=[]
    for state_path in (root/'footage').glob('*/recording.json'):
        state=metadata(state_path.parent)
        if state.get('ffmpeg_pid') and not state.get('stopped_utc') and alive(state['ffmpeg_pid']):
            active_recordings.append(str(state_path.parent))
    recording=bool(active_recordings)
    report = dict(confirmed=progress.get('confirmed_levels', 0), total=progress.get('total_levels', 76),
                  all_levels_confirmed=progress.get('all_levels_confirmed', False), recording=recording,
                  attempt=str(folder) if folder else None, stopped=result is not None,
                  level=row.get('expected_level') if row else None, board=row.get('board') if row else None,
                  visible_balls=row.get('balls') if row else None, elapsed=row.get('elapsed') if row else None,
                  gauntlet=attempt.get('gauntlet'),rank=row.get('hud') if row and attempt.get('gauntlet') else None,
                  gauntlet_sun_god_confirmed=len(gauntlet_boards),gauntlet_total=23,
                  recording_directories=active_recordings,result=result)
    print(json.dumps(report, ensure_ascii=True))
    return report


if __name__ == '__main__':
    status()
