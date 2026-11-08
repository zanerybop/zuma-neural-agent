import numpy as np
import torch
import pytest
from zuma_agent.board import Ball, normalize, to_window
from zuma_agent.ranker import TargetRanker, candidates, teacher_score
from zuma_agent.game_ui import GameUI


def test_saved_game_ready_dialog_is_navigation_not_gameplay():
    rows=[dict(text='GET READY!',confidence=.99,center=[320,135]),
          dict(text='GO',confidence=.99,center=[320,183])]
    assert GameUI.classify(rows)==('get_ready',[320,183])
    assert GameUI.classify(rows[1:])[0]=='unknown'


@pytest.mark.parametrize('text,rank,round_number', [('JAG I',2,1),('SUN 1',3,1),('SUN GOD 2',3,2),
                                                  ('SG 2',3,2),('司SGI',3,1),('RAB7+60',0,7),('SG1+100',3,1),
                                                  ('EAG7',1,7),('RABBIT 3',0,3),('LVL 13-1',None,None)])
def test_gauntlet_rank_is_not_confused_with_campaign_header(text,rank,round_number):
    ui=GameUI.__new__(GameUI)
    ui.read=lambda image:[dict(text=text)]
    hud=ui.gauntlet_hud(np.zeros((480,640,3),np.uint8))
    assert (hud['rank'],hud['round'])==(rank,round_number)


def test_playfield_coordinates_match_letterboxing():
    rgb = np.zeros((1440, 2560, 3), np.uint8)
    rgb[:, 320:2240] = (10, 20, 30)
    normalized = normalize(rgb)
    assert normalized.shape == (480, 640, 3)
    assert (normalized == (10, 20, 30)).all()
    assert to_window((320, 240), rgb.shape) == (.5, .5)
    assert to_window((0, 0), rgb.shape) == (.125, 0)


def test_front_ball_blocks_shot_to_rear_ball():
    balls = [Ball(200, 100, 0), Ball(300, 100, 1)]
    x, _ = candidates(balls, np.array([100., 100.]), 1)
    assert x[0, 4] == 0 and x[1, 4] == 1
    assert x[1, 0] == 1
    assert teacher_score(x)[0] > teacher_score(x)[1]


def test_matching_pair_preferred_over_single_same_color():
    balls = [Ball(100, 100, 0), Ball(132, 100, 0), Ball(300, 100, 0)]
    x, _ = candidates(balls, np.array([200., 300.]), 0)
    assert x[0, 1] == .4 and x[2, 1] == .2
    assert teacher_score(x)[0] > teacher_score(x)[2]


def test_ui_does_not_claim_win_from_points_alone():
    rows = [dict(text='POINTS', confidence=.99, center=[100,100]), dict(text='OK', confidence=.99, center=[320,390])]
    assert GameUI.classify(rows)[0] == 'unknown'
    rows.append(dict(text='STATS', confidence=.99, center=[320,160]))
    assert GameUI.classify(rows) == ('level_won', [320,390])


def test_ranker_scores_arbitrary_number_of_targets():
    assert TargetRanker()(torch.zeros((23,10))).shape == (23,)


@pytest.mark.parametrize('text,expected', [('IVL I-2', [1,2]), ('LVL 11-3', [11,3]), ('LVL H-I', [1,1]), ('I-5', [1,5]), ('LVL I8-6', None)])
def test_hud_ocr_does_not_treat_prefix_as_level_digits(text, expected):
    ui = GameUI.__new__(GameUI)
    ui.read = lambda image: [dict(text=text)]
    assert ui.hud(np.zeros((480,640,3), np.uint8))['level'] == expected


def test_motion_estimates_forward_and_backward_chain_speed():
    from zuma_agent.motion import ChainMotion
    motion = ChainMotion()
    curves = [np.array([[float(x), 100.] for x in range(500)])]
    def frame(delta):
        return [Ball(x+delta, 100., color, progress=(x+delta)/499) for color, x in enumerate([100.,150.,200.])]
    motion.update('test', frame(0), curves, 0)
    assert abs(motion.update('test', frame(10), curves, .1)[0]-100) < .01
    motion = ChainMotion()
    motion.update('test', frame(0), curves, 0)
    assert abs(motion.update('test', frame(-5), curves, .1)[0]+50) < .01


def test_ammo_separates_loaded_ball_and_rear_gem():
    import cv2
    from zuma_agent.board import BoardVision
    vision = BoardVision.__new__(BoardVision)
    vision.boards = {'test': dict(shooter=np.array([320.,240.]))}
    image = np.zeros((480,640,3), np.uint8)
    cv2.circle(image, (320,213), 11, (255,0,0), -1)
    cv2.circle(image, (320,264), 5, (0,255,0), -1)
    assert vision.ammo(image, 'test', [320.,100.]) == (2,3)
    image[195:230] = (180,150,200)  # Pale empty tongue is not a white ball.
    assert vision.ammo(image, 'test', [320.,100.])[0] is None
    image[195:230] = 0
    cv2.circle(image, (320,213), 11, (190,180,160), -1)
    assert vision.ammo(image, 'test', [320.,100.])[0] == 5


def test_full_video_export_refuses_incomplete_campaign(tmp_path, monkeypatch):
    import zuma_agent.video as video
    monkeypatch.setattr(video, 'index', lambda runs: dict(all_levels_confirmed=False, confirmed_levels=11, total_levels=76))
    with pytest.raises(RuntimeError, match='11/76'):
        video.export(output=tmp_path/'video')
    assert not (tmp_path/'video').exists()


def test_teacher_prioritizes_immediate_clear_over_building_a_single():
    head = np.array([1,.2,.85,.4,0,0,1,1,1,.5], dtype=np.float32)
    far_triple = np.array([1,.6,.2,.4,0,1,1,0,1,.5], dtype=np.float32)
    assert teacher_score(far_triple) > teacher_score(head)+5
    blocked_head = head.copy(); blocked_head[4] = 1
    assert teacher_score(blocked_head) < teacher_score(np.zeros(10, dtype=np.float32))


def test_collision_check_rejects_ray_that_lead_moves_into_other_color():
    from zuma_agent.trajectory import shot_geometry, choose_feasible
    balls = [Ball(200,0,0,progress=.5,track=1), Ball(100,40,2,progress=.5,track=0)]
    curves = [np.array([[0.,40.],[100.,40.],[200.,40.]]),
              np.array([[200.,-100.],[200.,0.],[200.,100.]])]
    features, _ = candidates(balls,np.array([0.,0.]),0)
    assert features[0,4] == 0  # Original center ray is clear.
    aims, hits, _ = shot_geometry(balls,[0.,0.],curves,{0:0,1:120})
    assert aims[0,1] > 40 and hits[0] == 1  # Led ray intersects the red blocker.


def test_collision_check_keeps_clear_matching_target():
    import json
    from zuma_agent.trajectory import shot_geometry, choose_feasible
    balls = [Ball(200,0,0,progress=.5),Ball(100,80,2,progress=.5)]
    curves = [np.array([[0.,0.],[100.,0.],[200.,0.]])]
    features,_ = candidates(balls,np.array([0.,0.]),0)
    _,hits,_ = shot_geometry(balls,[0.,0.],curves,{0:0})
    chosen,safe = choose_feasible(np.array([10.,1.]),features,balls,hits,0)
    assert chosen == 0 and safe
    assert json.loads(json.dumps(dict(index=chosen))) == dict(index=0)


def test_motion_does_not_alias_neighboring_equal_color_balls():
    from zuma_agent.motion import ChainMotion
    motion = ChainMotion()
    curves = [np.array([[float(x),100.] for x in range(500)])]
    def frame(delta):
        return [Ball(x+delta,100.,0,progress=(x+delta)/499) for x in [100.,132.,164.]]
    motion.update('test',frame(0),curves,0)
    motion.speeds[0] = 100.
    assert abs(motion.update('test',frame(20),curves,.2)[0]-100) < .01


def test_intercepting_same_color_single_does_not_inherit_rear_pair_score():
    from zuma_agent.trajectory import shot_geometry,choose_feasible
    balls=[Ball(300,0,0,progress=.5,track=1),Ball(300,32,0,progress=.52,track=1),
           Ball(150,40,0,progress=.1,track=0)]
    curves=[np.array([[0.,40.],[150.,40.],[300.,40.]]),
            np.array([[300.,-100.],[300.,0.],[300.,100.]])]
    features,_=candidates(balls,np.array([0.,0.]),0)
    assert features[0,4]==0
    _,hits,_=shot_geometry(balls,[0.,0.],curves,{0:0,1:120})
    assert hits[0]==2
    chosen,safe=choose_feasible(np.array([10.,9.,1.]),features,balls,hits,0)
    assert chosen==2 and safe


def test_critical_single_can_outrank_pair_immediately_behind_it():
    front=np.array([1,.2,.969,.3,0,0,1,1,1,.5],dtype=np.float32)
    rear_pair=np.array([1,.4,.945,.3,0,.5,1,1,1,.5],dtype=np.float32)
    assert teacher_score(front)>teacher_score(rear_pair)+4


@pytest.mark.parametrize('width,scale',[(32,1.),(64,20.)])
def test_ranker_loader_preserves_legacy_and_scaled_model_outputs(width,scale):
    from zuma_agent.ranker import load_ranker
    original=TargetRanker(width=width,output_scale=scale)
    restored=load_ranker(dict(model=original.state_dict(),width=width,output_scale=scale))
    x=torch.rand(4,10)
    assert torch.allclose(original(x),restored(x))
