"""Recorded, bounded live campaign attempts with a learned target ranker."""
import argparse
import hashlib
from datetime import datetime, timezone
import json
import os
import shutil
import re
from pathlib import Path
import time
import numpy as np
import torch
from PIL import Image
from mss import MSS
from .board import BoardVision, COLORS, DEFAULT_GAME, normalize, to_window, from_window
from .footage import mark
from .ranker import load_ranker, candidates, teacher_score
from .vision import capture, capture_ammo
from .windows import GameWindow, down, user32
from .game_ui import GameUI
from .motion import ChainMotion
from .trajectory import shot_geometry, choose_feasible, discard_aim
from .gun_clock import GunClock


def keypress(key):
    user32.keybd_event(key, 0, 0, 0)
    time.sleep(.04)
    user32.keybd_event(key, 0, 2, 0)


def run(args):
    gauntlet=getattr(args,'gauntlet',None)
    recording_folder=os.environ.get('ZUMA_FOOTAGE_DIR','footage/current')
    output = Path(args.output)
    if (output/'attempt.json').exists():
        raise ValueError('Use a new output directory; existing attempt evidence must be preserved.')
    output.mkdir(parents=True, exist_ok=True)
    vision = BoardVision(args.game)
    if args.stage not in range(1, 14) or not 1 <= args.level <= len(vision.stages[f'stage{args.stage}'].split(',')):
        raise ValueError('Stage or level is outside the installed campaign.')
    checkpoint = torch.load(args.checkpoint, weights_only=True, map_location='cpu')
    if checkpoint.get('kind') != 'target_ranker':
        raise ValueError('Expected target_ranker checkpoint')
    model = load_ranker(checkpoint)
    torch.set_num_threads(2)
    window = GameWindow.find()
    if not window.focused():
        raise RuntimeError('Focus the game first. No automatic focus stealing.')
    mark(recording_folder, f'Neural campaign attempt started. Checkpoint {args.checkpoint}; training source {checkpoint["source"]}.')
    attempt = dict(started_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
        checkpoint=args.checkpoint, policy='neural_target_ranker', training_source=checkpoint['source'],
        adaptive_lead=args.adaptive_lead, precise_ammo=args.precise_ammo, collision_checks=True, post_aim_ammo_check=True,
        checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        seconds=args.seconds, cooldown=args.cooldown, gun_cycle_min=max(.24,args.cooldown),
        debug_shots=args.debug_shots, restart_on_game_over=args.restart_on_game_over,
        max_restarts=args.max_restarts, gauntlet=gauntlet, recording_folder=recording_folder, status='running')
    (output/'attempt.json').write_text(json.dumps(attempt, indent=2), encoding='utf-8')
    (output/'source').mkdir(exist_ok=True)
    for source in Path(__file__).parent.glob('*.py'):
        shutil.copyfile(source, output/'source'/source.name)
    shutil.copyfile(args.checkpoint, output/'policy.pt')
    ui = GameUI()
    if args.resume:
        with MSS() as preflight:
            if ui.is_paused(normalize(capture(preflight,window))):
                keypress(0x20)
                time.sleep(.3)
    attempt['requested_utc']=attempt['started_utc']
    attempt['started_utc']=datetime.now(timezone.utc).isoformat()
    (output/'attempt.json').write_text(json.dumps(attempt,indent=2),encoding='utf-8')
    started=time.monotonic()
    motion = ChainMotion()
    last_ui, last_hud, last_win, last_navigation = -100., -100., -100., 0.
    stage, level = args.stage, args.level
    victories, hud, ui_phase = [], None, 'unverified'
    mismatch_value, mismatch_count = None, 0
    restarts,recovering = 0,False
    expected_board = lambda: (name if gauntlet=='random' else gauntlet) if gauntlet else vision.stages['stage'+str(stage)].split(',')[level-1]
    sun_observations=0
    last_save,last_print = -100.,-100.
    gun_clock=GunClock(max(.24,args.cooldown))
    preparation_estimate=.10
    accuracy_until=0.
    shots, ticks, feature_buffer, shard, debug_count = 0, 0, [], 0, 0
    reason = 'time_limit'

    def flush():
        nonlocal shard
        if feature_buffer:
            features = np.concatenate(feature_buffer)
            np.savez_compressed(output/f'candidates-{shard:05d}.npz', features=features,
                                teacher_scores=teacher_score(features), source='real_perception_teacher_labels')
            feature_buffer.clear(); shard += 1

    try:
        with MSS() as sct, (output/'decisions.jsonl').open('a', encoding='utf-8') as log, torch.inference_mode():
            while time.monotonic()-started < args.seconds:
                tick = time.monotonic(); elapsed = tick-started
                if down(0x78) or down(0x1B) or (output/'STOP').exists():
                    reason = 'stop_requested'; break
                if not window.focused():
                    reason = 'focus_lost'; break
                raw = capture(sct, window)
                observed_at = time.monotonic()
                image = normalize(raw)
                if ui.is_paused(image):
                    reason='game_paused'; ui_phase='paused'; break
                name, error = vision.identify(image)
                panel = ui.panel_hint(image)
                # OCR is reserved for transitions; the fast perception loop keeps
                # playing at full speed while a recognized board is active.
                if (error >= 20 or panel or ui.ready_hint(image)) and elapsed-last_ui > 2:
                    rows = ui.read(image)
                    ui_phase, button = ui.classify(rows)
                    last_ui = elapsed
                    transition = dict(elapsed=elapsed, phase=ui_phase, ocr=rows, expected=[stage, level])
                    with (output/'transitions.jsonl').open('a', encoding='utf-8') as transitions:
                        transitions.write(json.dumps(transition)+'\n')
                    print(json.dumps(dict(elapsed=round(elapsed, 1), phase=ui_phase, level=[stage,level])), flush=True)
                    if ui_phase == 'level_won':
                        if gauntlet:
                            Image.fromarray(image).save(output/f'gauntlet-stats-{int(elapsed):05d}.png')
                            window.act(to_window(button,raw.shape),1)
                            last_navigation=elapsed
                            time.sleep(.7)
                            continue
                        if elapsed-last_win < 5:
                            window.act(to_window(button, raw.shape), 1)
                            time.sleep(.3)
                            continue
                        last_win = elapsed
                        evidence = output/f'victory-{stage:02d}-{level:02d}-pass-{restarts:03d}.png'
                        Image.fromarray(image).save(evidence)
                        victory = dict(stage=stage, level=level, board=expected_board(), elapsed=elapsed,
                                       evidence=str(evidence), ocr=rows, policy='neural_target_ranker')
                        victories.append(victory)
                        (output/'victories.json').write_text(json.dumps(victories, indent=2), encoding='utf-8')
                        from .progress import index
                        index()
                        mark(recording_folder, f'Level {stage}-{level} victory recorded; screenshot {evidence}.')
                        window.act(to_window(button, raw.shape), 1)
                        level += 1
                        if level > len(vision.stages['stage'+str(stage)].split(',')):
                            stage += 1; level = 1
                        if stage > 13:
                            reason = 'all_level_stats_observed'; break
                        time.sleep(1)
                        continue
                    if ui_phase in ('stage_menu', 'instructions', 'main_menu', 'chapter_dialogue', 'temple_confirmation','get_ready') and button:
                        if ui_phase=='stage_menu' and recovering:
                            for row in rows:
                                match=re.search(r'STAGE\s*(\d{1,2})',row['text'].upper())
                                if match and 1<=int(match[1])<=13:
                                    stage,level=int(match[1]),1
                                    break
                        last_navigation = elapsed
                        window.act(to_window(button, raw.shape), 1)
                        time.sleep(1)
                        continue
                    if ui_phase == 'game_over':
                        Image.fromarray(image).save(output/f'game-over-pass-{restarts:03d}.png')
                        if gauntlet:
                            reason='gauntlet_game_over'; break
                        if not args.restart_on_game_over:
                            reason='game_over'; break
                        if not recovering:
                            if restarts>=args.max_restarts:
                                reason='game_over_restart_limit'; break
                            restarts+=1; recovering=True
                            level=1; last_navigation=elapsed
                            gun_clock=GunClock(max(.24,args.cooldown))
                            motion=ChainMotion(); accuracy_until=0.
                            mark(recording_folder,f'Normal game-over recovery {restarts}: retrying stage {stage} from level 1. Earlier victory evidence is preserved.')
                        window.act(to_window(button if button else [320,240],raw.shape),1)
                        time.sleep(1)
                        continue
                    if ui_phase == 'paused':
                        reason = 'game_paused'; break
                if error < 20 and elapsed-last_hud > (2 if gauntlet else 8):
                    hud = ui.gauntlet_hud(image) if gauntlet else ui.hud(image); last_hud = elapsed
                    if gauntlet:
                        sun_observations=sun_observations+1 if hud['rank']==3 and hud['round'] and hud['round']>=1 else 0
                        if sun_observations>=2:
                            Image.fromarray(image).save(output/'sun-god-reached.png')
                            (output/'sun-god.json').write_text(json.dumps(dict(board=gauntlet,hud=hud,elapsed=elapsed),indent=2),encoding='utf-8')
                            reason='sun_god_reached'; break
                    if name != expected_board():
                        if elapsed-last_ui < 10:
                            time.sleep(.1)
                            continue
                        reason = f'board_mismatch_expected_{expected_board()}_saw_{name}'
                        Image.fromarray(image).save(output/'board-mismatch.png'); break
                    if not gauntlet and hud['level'] and hud['level'] != [stage, level] and elapsed-max(last_win, last_navigation) > 10:
                        mismatch_count = mismatch_count+1 if mismatch_value == hud['level'] else 1
                        mismatch_value = hud['level']
                        if mismatch_count >= 3:
                            reason = f'level_mismatch_{hud["level"]}_expected_{stage}_{level}'
                            Image.fromarray(image).save(output/'level-mismatch.png'); break
                    else:
                        mismatch_value, mismatch_count = None, 0
                if error < 20 and time.monotonic()-observed_at > .12:
                    # HUD OCR can take several game updates. Reacquire pixels
                    # before making a shot from a stale board observation.
                    raw = capture(sct,window); observed_at = time.monotonic()
                    image = normalize(raw)
                    name,error = vision.identify(image)
                    panel = ui.panel_hint(image)
                if error < 20 and name == expected_board():
                    ui_phase = 'playing'
                    if hud and hud.get('level')==[stage,level]:
                        recovering=False
                balls = vision.detect(image, name) if error < 20 and not panel else []
                speed = motion.update(name, balls, vision.boards[name]['curves'], observed_at)
                cursor = from_window(window.cursor(), raw.shape)
                if vision.accuracy_visible(image,name,cursor):
                    accuracy_until=time.monotonic()+.35
                projectile_speed=1500. if time.monotonic()<accuracy_until else 600.
                if args.precise_ammo:
                    color, next_color = vision.ammo(image, name, cursor)
                else:
                    color, next_color = vision.shooter_color(image, name), None
                features, aims = candidates(balls, vision.boards[name]['shooter'], color)
                action = None
                if len(features) and error < 20 and color is not None:
                    scores = model(torch.from_numpy(features)).numpy()
                    predicted_aims, first_hits, leads = shot_geometry(balls, vision.boards[name]['shooter'],
                        vision.boards[name]['curves'], speed, args.adaptive_lead,
                        latency=time.monotonic()-observed_at+preparation_estimate+.03,
                        projectile_speed=projectile_speed)
                    chosen, safe_shot = choose_feasible(scores, features, balls, first_hits, color)
                    swap = False
                    if next_color is not None and next_color != color:
                        alternate = features.copy()
                        alternate[:,0] = np.array([b.color == next_color for b in balls],dtype=np.float32)
                        future_scores = model(torch.from_numpy(alternate)).numpy()
                        future, future_safe = choose_feasible(future_scores, alternate, balls, first_hits, next_color)
                        swap = future_safe and (not safe_shot or float(future_scores[future]) > float(scores[chosen])+1.2)
                    # The network alone ranks targets. Geometric pre-processing and
                    # timing remain explicit parts of the controller.
                    if time.monotonic()+preparation_estimate >= gun_clock.ready_at() and features[chosen,4]==0:
                        if swap:
                            time.sleep(max(0,gun_clock.ready_at()-time.monotonic()))
                            window.act(to_window(cursor, raw.shape), 2)
                            gun_clock.swapped(time.monotonic())
                            action = dict(kind='swap', current=color, next=next_color)
                            log.write(json.dumps(dict(elapsed=elapsed, shot=action, expected_level=[stage,level]))+'\n'); log.flush()
                            time.sleep(.04)
                            continue
                        target = predicted_aims[chosen].copy()
                        discarding = False
                        if not safe_shot:
                            escape = discard_aim(balls, vision.boards[name]['shooter'])
                            if escape is not None:
                                target = escape
                                discarding = True
                        b = balls[chosen]
                        lead = float(leads[chosen])
                        prepare_started=time.monotonic()
                        window.act(to_window(target, raw.shape), 0)
                        time.sleep(max(.04,gun_clock.ready_at()-time.monotonic()))
                        if args.precise_ammo:
                            prepared = (normalize(capture(sct, window)) if debug_count < args.debug_shots
                                        else capture_ammo(sct,window,vision.boards[name]['shooter']))
                            actual_color, _ = vision.ammo(prepared, name, target)
                            if actual_color != color:
                                log.write(json.dumps(dict(elapsed=elapsed, expected_level=[stage,level],
                                    kind='aim_probe', expected_color=color, observed_color=actual_color,
                                    aim=target.tolist(), board=name))+'\n'); log.flush()
                                time.sleep(.02)
                                continue
                        debug_prefix = None
                        if debug_count < args.debug_shots and np.linalg.norm(target-vision.boards[name]['shooter']) > 180:
                            debug_count += 1
                            debug_prefix = output/f'debug-shot-{debug_count:03d}'
                            Image.fromarray(prepared if args.precise_ammo else image).save(str(debug_prefix)+'-before.png')
                            Path(str(debug_prefix)+'.json').write_text(json.dumps(dict(elapsed=elapsed, board=name,
                                origin=vision.boards[name]['shooter'].tolist(), aim=target.tolist(),
                                color=color, target_index=chosen, lead=lead)), encoding='utf-8')
                        gun_clock.wait_until_ready()
                        fired_at = time.monotonic()
                        if not window.trigger(1):
                            continue
                        gun_clock.fired(fired_at)
                        preparation_estimate=float(np.clip(.8*preparation_estimate+.2*(fired_at-prepare_started),.06,.3))
                        shots += 1
                        if debug_prefix:
                            time.sleep(.10)
                            Image.fromarray(normalize(capture(sct,window))).save(str(debug_prefix)+'-after.png')
                        action = dict(kind='discard' if discarding else 'fire', aim=target.tolist(), index=chosen, score=float(scores[chosen]), lead=lead,
                                      observation_age=fired_at-observed_at,
                                      collision_verified=safe_shot, predicted_first_hit=int(first_hits[chosen]),
                                      match=bool(features[chosen, 0]), group=float(features[chosen, 1])*5)
                    feature_buffer.append(features)
                    if len(feature_buffer) >= 256:
                        flush()
                row = dict(elapsed=elapsed, board=name, board_error=error, balls=len(balls), expected_level=[stage, level], hud=hud,
                           shooter=COLORS[color] if color is not None else None, next_color=next_color, chain_speed=speed,
                           projectile_speed=projectile_speed,
                           shot=action, policy='neural_target_ranker')
                log.write(json.dumps(row)+'\n'); log.flush()
                if elapsed-last_save > 10 or (not balls and elapsed-last_save > 3):
                    Image.fromarray(image).save(output/f'frame-{int(elapsed):05d}.png')
                    Image.fromarray(vision.annotate(image, name, balls)).save(output/'latest-detection.png')
                    last_save = elapsed
                if elapsed-last_print > 5:
                    print(json.dumps(dict(elapsed=round(elapsed, 1), board=name, error=round(error, 1), balls=len(balls),
                                          shooter=COLORS[color] if color is not None else None, shots=shots)), flush=True)
                    last_print = elapsed
                ticks += 1
                time.sleep(max(0, .08-(time.monotonic()-tick)))
    except KeyboardInterrupt:
        reason = 'keyboard_interrupt'
    except Exception as error:
        reason = f'error_{type(error).__name__}: {error}'
        raise
    finally:
        flush()
        # Leave a bounded attempt paused so the board cannot lose unattended lives.
        if window.focused() and ui_phase not in ('level_won', 'game_over', 'paused', 'stage_menu', 'chapter_dialogue', 'temple_confirmation'):
            keypress(0x20)
        from .progress import index
        verified = index(game=args.game)
        result = dict(stopped_utc=datetime.now(timezone.utc).isoformat(), reason=reason, shots=shots,
                      seconds=time.monotonic()-started, ticks=ticks, status='attempt_finished',
                      campaign_complete=verified['all_levels_confirmed'], victories=len(victories),
                      next_level=[stage, level], restarts=restarts, gauntlet=gauntlet,
                      sun_god_reached=bool(gauntlet and reason=='sun_god_reached'))
        (output/'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        attempt.update(result)
        (output/'attempt.json').write_text(json.dumps(attempt, indent=2), encoding='utf-8')
        mark(recording_folder, f'Neural attempt finished: {json.dumps(result)}')
        print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='runs/campaign-001')
    p.add_argument('--checkpoint', default='models/target-ranker.pt')
    p.add_argument('--game', default=str(DEFAULT_GAME))
    p.add_argument('--seconds', type=float, default=180)
    p.add_argument('--cooldown', type=float, default=.23)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stage', type=int, default=1)
    p.add_argument('--level', type=int, default=1)
    p.add_argument('--adaptive-lead', action='store_true')
    p.add_argument('--precise-ammo', action='store_true')
    p.add_argument('--debug-shots', type=int, default=0)
    p.add_argument('--restart-on-game-over',action='store_true')
    p.add_argument('--max-restarts',type=int,default=8)
    p.add_argument('--gauntlet',default=None,help='Gauntlet board id, or random; stop after confirmed Sun God rank')
    args = p.parse_args()
    if args.seconds <= 0 or args.cooldown < .1:
        p.error('seconds must be positive; cooldown must be at least .1')
    run(args)


if __name__ == '__main__':
    main()
