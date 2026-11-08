"""Earn Gauntlet ranks through ordinary menus and the neural campaign controller."""
import argparse
from argparse import Namespace
from datetime import datetime, timezone
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import time
import cv2
import numpy as np
from PIL import Image
from mss import MSS
from .board import BoardVision,DEFAULT_GAME,normalize,to_window
from .campaign import run
from .game_ui import GameUI
from .vision import capture
from .windows import GameWindow,down


def confirmed(root=Path('runs')):
    return {json.loads(p.read_text())['board'] for p in root.glob('gauntlet-*/sun-god.json')
            if (p.parent/'sun-god-reached.png').exists()}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--game',default=str(DEFAULT_GAME))
    parser.add_argument('--checkpoint',default='models/target-ranker.pt')
    parser.add_argument('--output',default='runs/gauntlet-session-001')
    parser.add_argument('--seconds',type=float,default=21600)
    parser.add_argument('--from-game',action='store_true',help='Return from a paused live board to Gauntlet selection first')
    args=parser.parse_args()
    folder=Path(args.output)
    folder.mkdir(parents=True,exist_ok=False)
    ui=GameUI(); vision=BoardVision(args.game); window=GameWindow.find()
    boards=list(dict.fromkeys(b for stage in range(1,14) for b in vision.stages[f'stage{stage}'].split(',')))
    boards.append('random')
    xml=(vision.root/'levels/levels.xml').read_text()
    names={}
    for item in re.findall(r'<Graphics\b([^>]*)>',xml):
        meta=dict(re.findall(r'(\w+)\s*=\s*"([^"]*)"',item))
        names[meta['id']]=meta.get('dispname',meta['id'])
    names['random']='Random'
    clean=lambda s:re.sub('[^A-Z0-9]','',s.upper())
    counter=max([int(p.name.split('-')[1]) for p in Path('runs').glob('gauntlet-[0-9]*')]+[0])
    started=time.monotonic(); events=[]; shot_id=0
    reason='time_limit'
    with MSS() as sct:
        def check():
            if down(0x78) or down(0x1B) or (folder/'STOP').exists():
                raise InterruptedError('stop_requested')
            if not window.focused():
                raise InterruptedError('focus_lost')
        def snapshot(label):
            nonlocal shot_id
            check(); raw=capture(sct,window); im=normalize(raw)
            shot_id+=1; Image.fromarray(im).save(folder/f'ui-{shot_id:04d}-{label}.png')
            return raw,im
        def click(point,delay=.4):
            check(); raw=capture(sct,window); window.act(to_window(point,raw.shape),1); time.sleep(delay)
        def note(kind,**values):
            event=dict(utc=datetime.now(timezone.utc).isoformat(),kind=kind,**values)
            events.append(event)
            (folder/'events.json').write_text(json.dumps(events,indent=2),encoding='utf-8')
            print(json.dumps(event),flush=True)
        def menu_board():
            _,im=snapshot('selection')
            text=''.join(r['text'] for r in ui.read(cv2.resize(im[402:439,175:480],(610,74))))
            scores={b:SequenceMatcher(None,clean(text),clean(names[b])).ratio() for b in boards}
            board=max(scores,key=scores.get)
            if scores[board]<.7:
                raise RuntimeError(f'Unknown Gauntlet map: {text!r}, scores={scores}')
            return board
        def return_to_selection():
            # The campaign controller pauses live gameplay on return. The MENU
            # button opens options even while the board is paused.
            click((578,14)); _,im=snapshot('options'); rows=ui.read(im)
            button=next((r['center'] for r in rows if 'BACKTOMAINMENU' in clean(r['text'])),None)
            if not button:
                raise RuntimeError('Expected gameplay options with BACK TO MAIN MENU')
            click(button); _,im=snapshot('leave-confirmation'); rows=ui.read(im)
            button=next((r['center'] for r in rows if clean(r['text'])=='LEAVE'),None)
            if not button:
                raise RuntimeError('Expected LEAVE confirmation')
            click(button,.7); _,im=snapshot('main-menu'); rows=ui.read(im)
            button=next((r['center'] for r in rows if clean(r['text'])=='GAUNTLET'),None)
            if not button:
                raise RuntimeError('Expected main menu GAUNTLET')
            click(button,.6); _,im=snapshot('continue-or-selection'); rows=ui.read(im)
            button=next((r['center'] for r in rows if clean(r['text'])=='NEWGAME'),None)
            if button:
                click(button,.6)
        try:
            if args.from_game:
                return_to_selection()
            while time.monotonic()-started<args.seconds:
                check(); done=confirmed()
                (folder/'progress.json').write_text(json.dumps(dict(required=boards,confirmed=sorted(done),
                    remaining=[b for b in boards if b not in done]),indent=2),encoding='utf-8')
                if set(boards)<=done:
                    click((80,445),.7)
                    _,final_image=snapshot('hat-verification')
                    Image.fromarray(final_image).save(folder/'hat-verification.png')
                    reason='all_gauntlet_sun_god_confirmed'; break
                board=menu_board()
                if board in done:
                    click((455,460)); continue
                # Only unlocked ranks respond. Select the highest available tier.
                click((320,145),.15); click((320,100),.15)
                snapshot('rank-selected')
                click((320,460),.7)
                counter+=1
                output=Path(f'runs/gauntlet-{counter:03d}-{board}')
                note('attempt_started',board=board,output=str(output),confirmed=len(done),total=len(boards))
                run(Namespace(output=str(output),game=str(vision.root),stage=1,level=1,
                    checkpoint=args.checkpoint,adaptive_lead=True,precise_ammo=True,
                    seconds=min(2400,args.seconds-(time.monotonic()-started)),cooldown=.24,
                    debug_shots=0,restart_on_game_over=False,max_restarts=0,resume=True,gauntlet=board))
                result=json.loads((output/'result.json').read_text())
                note('attempt_finished',board=board,result=result)
                if result['reason'] in ('stop_requested','focus_lost','game_paused') or result['reason'].startswith('error_'):
                    reason=result['reason']; break
                if result['reason']=='gauntlet_game_over':
                    # Preserve the loss screen; let the user-facing game's own
                    # OK button return to selection before retrying this map.
                    button=None
                    for poll in range(10):
                        _,im=snapshot('game-over'); rows=ui.read(im)
                        button=next((r['center'] for r in rows if clean(r['text']) in ('OK','CONTINUE','TRYAGAIN')),None)
                        if button:
                            break
                        time.sleep(.5)
                    if not button:
                        raise RuntimeError('Unrecognized Gauntlet Game Over navigation')
                    click(button,.7)
                else:
                    return_to_selection()
                if board in confirmed():
                    note('sun_god_saved',board=board,confirmed=len(confirmed()))
                    click((455,460))
            note('session_finished',reason=reason,confirmed=len(confirmed()))
        except Exception as error:
            reason=f'{type(error).__name__}: {error}'
            note('session_stopped',reason=reason,confirmed=len(confirmed()))
    (folder/'result.json').write_text(json.dumps(dict(reason=reason,confirmed=sorted(confirmed())),indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
