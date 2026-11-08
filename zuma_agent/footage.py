"""Persistent desktop recording, independent of the agent and training process."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def stamp():
    return datetime.now(timezone.utc).isoformat()


def mark(folder, message):
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'events.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(dict(utc=stamp(), message=message), ensure_ascii=False)+'\n')


def alive(pid):
    # Query via Windows process handles without sending termination signals.
    import ctypes as C
    kernel = C.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [C.c_ulong, C.c_int, C.c_ulong]
    kernel.OpenProcess.restype = C.c_void_p
    kernel.GetExitCodeProcess.argtypes = [C.c_void_p, C.POINTER(C.c_ulong)]
    kernel.CloseHandle.argtypes = [C.c_void_p]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    try:
        code = C.c_ulong()
        return bool(kernel.GetExitCodeProcess(handle, C.byref(code))) and code.value == 259
    finally:
        kernel.CloseHandle(handle)


def metadata(folder):
    path = folder/'recording.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def worker(folder, fps, encoder):
    folder = Path(folder).resolve()
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('ffmpeg is not installed')
    folder.mkdir(parents=True, exist_ok=True)
    options = ['-c:v', 'h264_nvenc', '-preset', 'p4', '-cq', '25', '-b:v', '0'] if encoder == 'nvenc' else [
        '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23']
    cmd = [ffmpeg, '-hide_banner', '-y', '-f', 'gdigrab', '-framerate', str(fps),
           '-draw_mouse', '1', '-i', 'desktop', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
           *options, '-pix_fmt', 'yuv420p', '-g', str(fps*2), '-f', 'segment',
           '-segment_time', '300', '-reset_timestamps', '1', '-strftime', '1',
           str(folder/'screen-%Y%m%d-%H%M%S.mkv')]
    with (folder/'ffmpeg.log').open('a', encoding='utf-8') as log:
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=log, stderr=log,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        state = dict(worker_pid=os.getpid(), ffmpeg_pid=process.pid, started_utc=stamp(),
                     fps=fps, encoder=encoder, audio=False, directory=str(folder), command=cmd)
        (folder/'recording.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
        mark(folder, 'Desktop recording started (screen and cursor, no audio).')
        while process.poll() is None and not (folder/'STOP').exists():
            time.sleep(.5)
        if process.poll() is None:
            process.communicate(input=b'q\n', timeout=30)
        state.update(stopped_utc=stamp(), exit_code=process.returncode)
        (folder/'recording.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
        mark(folder, f'Recording stopped; ffmpeg exit code {process.returncode}.')


def main():
    parser = argparse.ArgumentParser(description='Record project footage for later editing')
    parser.add_argument('command', choices=['start', 'status', 'mark', 'stop', 'worker'])
    parser.add_argument('--dir', default='footage/current')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--encoder', choices=['nvenc', 'cpu'], default='nvenc')
    parser.add_argument('--message', default='')
    args = parser.parse_args()
    if args.fps < 1 or args.fps > 60:
        parser.error('fps must be 1..60')
    folder = Path(args.dir).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    state = metadata(folder)
    running = bool(state.get('worker_pid') and not state.get('stopped_utc') and alive(state['worker_pid']))
    if args.command == 'worker':
        worker(folder, args.fps, args.encoder)
    elif args.command == 'start':
        if running:
            print(f'Already recording: {folder}')
            return
        if not shutil.which('ffmpeg'):
            parser.error('ffmpeg not found')
        (folder/'STOP').unlink(missing_ok=True)
        with (folder/'worker.log').open('a', encoding='utf-8') as log:
            process = subprocess.Popen([sys.executable, '-m', 'zuma_agent.footage', 'worker',
                                        '--dir', str(folder), '--fps', str(args.fps), '--encoder', args.encoder],
                                       cwd=Path(__file__).resolve().parent.parent,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                       creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS)
        time.sleep(2)
        state = metadata(folder)
        if process.poll() is not None or not state.get('ffmpeg_pid') or not alive(state['ffmpeg_pid']):
            parser.exit(1, f'Recorder failed. Inspect {folder / "ffmpeg.log"} and worker.log\n')
        print(f'Recording started: {folder}; worker PID {process.pid}')
    elif args.command == 'status':
        files = list(folder.glob('screen-*.mkv'))
        print(json.dumps(dict(running=running, files=len(files),
                              bytes=sum(p.stat().st_size for p in files), metadata=state), indent=2))
    elif args.command == 'mark':
        if not args.message:
            parser.error('Provide --message')
        mark(folder, args.message)
    elif args.command == 'stop':
        if not running:
            print('No active recorder in this directory.')
            return
        (folder/'STOP').touch()
        deadline = time.monotonic()+15
        while alive(state['worker_pid']) and time.monotonic() < deadline:
            time.sleep(.25)
        if alive(state['worker_pid']):
            parser.exit(1, f'Stop requested; recorder still finalizing. Check {folder / "ffmpeg.log"}\n')
        print(f'Recording saved: {folder}')


if __name__ == '__main__':
    main()
