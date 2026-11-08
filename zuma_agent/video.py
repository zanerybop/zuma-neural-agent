"""Export a verified full playthrough from the continuous desktop recording."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
from .progress import index
from .footage import metadata, alive


def duration(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', str(path)],
                            check=True, capture_output=True, text=True)
    value = json.loads(result.stdout).get('format', {}).get('duration')
    if value:
        return float(value)
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                             'packet=pts_time', '-of', 'csv=p=0', str(path)],
                            check=True, capture_output=True, text=True)
    values = [float(line.strip().strip(',')) for line in result.stdout.splitlines() if line.strip().strip(',')]
    return max(values)+1/30 if values else 0


def export(recording='footage/current', output='footage/final', runs='runs'):
    progress = index(runs)
    if not progress['all_levels_confirmed']:
        raise RuntimeError(f"Full playthrough is not yet verified: {progress['confirmed_levels']}/{progress['total_levels']} levels. No full-playthrough video was exported.")
    recording, output, runs = Path(recording).resolve(), Path(output).resolve(), Path(runs)
    state = metadata(recording)
    sources = sorted(recording.glob('screen-*.mkv'))
    if not state or not sources:
        raise RuntimeError('Recording metadata or video segments are missing')
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        raise RuntimeError('FFmpeg and FFprobe are required')
    output.mkdir(parents=True, exist_ok=True)
    worker_alive = bool(not state.get('stopped_utc') and state.get('worker_pid') and alive(state['worker_pid']))
    if worker_alive:
        snapshot = output/'active-segment-snapshot.mkv'
        shutil.copyfile(sources[-1], snapshot)
        sources[-1] = snapshot
    entries = []
    for path in sources:
        seconds = duration(path)
        if seconds > 0:
            entries.append((path, seconds))
    attempts = [json.loads(p.read_text(encoding='utf-8')) for p in runs.glob('campaign-*/attempt.json')]
    start = min(datetime.fromisoformat(a['started_utc']) for a in attempts)
    last_win = max(datetime.fromisoformat(json.loads((Path(row['evidence']['attempt'])/'attempt.json').read_text(encoding='utf-8'))['started_utc']).timestamp()+row['evidence']['elapsed']
                   for row in progress['levels'])
    recording_start = datetime.fromisoformat(state['started_utc']).timestamp()
    offset = max(0, start.timestamp()-recording_start-3)
    # Keep the final victory animation and the full congratulations message.
    length = last_win-recording_start-offset+45
    if sum(seconds for _, seconds in entries) < offset+length-2:
        raise RuntimeError('Recording does not yet contain the end of the final victory. Retry in a few seconds.')
    concat = output/'segments.txt'
    with concat.open('w', encoding='utf-8') as stream:
        for path, seconds in entries:
            escaped = path.as_posix().replace("'", "'\\''")
            stream.write(f"file '{escaped}'\nduration {seconds:.6f}\n")
    target = output/'neural-playthrough.mkv'
    cmd = ['ffmpeg', '-hide_banner', '-n', '-ss', str(offset), '-f', 'concat', '-safe', '0', '-i', str(concat),
           '-t', str(length), '-c', 'copy', '-avoid_negative_ts', 'make_zero', str(target)]
    with (output/'export.log').open('w', encoding='utf-8') as log:
        subprocess.run(cmd, stdout=log, stderr=log, check=True)
    report = dict(output=str(target), source_segments=[dict(path=str(p), seconds=s) for p,s in entries],
                  offset_seconds=offset, duration_seconds=length, confirmed_levels=progress['confirmed_levels'],
                  all_levels_confirmed=True, audio=False, contains_debugging_and_failed_attempts=True,
                  attempts=attempts, levels=progress['levels'])
    (output/'playthrough-manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (output/'chapters.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['level','board','victory_video_seconds','victory_utc','screenshot','checkpoint'])
        for row in progress['levels']:
            evidence=row['evidence']
            attempt=json.loads((Path(evidence['attempt'])/'attempt.json').read_text(encoding='utf-8'))
            victory=datetime.fromisoformat(attempt['started_utc']).timestamp()+evidence['elapsed']
            writer.writerow([f"{row['stage']}-{row['level']}",row['board'],round(victory-recording_start-offset,3),
                             datetime.fromtimestamp(victory,datetime.fromisoformat(attempt['started_utc']).tzinfo).isoformat(),
                             evidence['evidence'],attempt.get('checkpoint','')])
    print(f'Full verified playthrough: {target}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--recording', default='footage/current')
    p.add_argument('--output', default='footage/final')
    p.add_argument('--runs', default='runs')
    args = p.parse_args()
    try:
        export(args.recording, args.output, args.runs)
    except (RuntimeError, subprocess.CalledProcessError) as error:
        p.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
