import cv2
import numpy as np
import pytest
from zuma_agent.gun_clock import GunClock
from zuma_agent.board import BoardVision,Ball
from zuma_agent.trajectory import shot_geometry
from zuma_agent.trajectory import discard_aim


def test_scrolling_nebula_is_recognized_without_becoming_purple_balls():
    vision=BoardVision.__new__(BoardVision)
    curve=np.array([[x,240.] for x in range(30,611,3)],np.float32)
    mask=np.zeros((480,640),np.uint8)
    cv2.polylines(mask,[curve.astype(np.int32)],False,255,52)
    vision.boards={'space':dict(background=np.zeros((480,640,3),np.uint8),
                               small=np.zeros((55,80,3),np.uint8),fixed=np.ones((55,80),bool),
                               curves=[curve],mask=mask,shooter=np.array([320.,400.]))}
    image=np.full((480,640,3),(100,16,110),np.uint8)
    cv2.circle(image,(200,240),16,(20,150,255),-1)
    assert vision.identify(image)==('space',8.)
    balls=vision.detect(image,'space')
    assert len(balls)==1 and balls[0].color==0


def test_unusable_ammo_exits_through_gap_instead_of_adding_chain_ball():
    origin=np.array([320.,240.])
    balls=[Ball(320+100*np.cos(a),240+100*np.sin(a),0)
           for a in np.linspace(0,2*np.pi,36,endpoint=False) if not .4<a<1.8]
    aim=discard_aim(balls,origin)
    assert aim is not None and aim[1]>400
    balls=[Ball(320+100*np.cos(a),240+100*np.sin(a),0)
           for a in np.linspace(0,2*np.pi,36,endpoint=False)]
    assert discard_aim(balls,origin) is None


def test_reload_window_is_measured_from_actual_fire_not_observation():
    gun=GunClock()
    gun.fired(1.)
    with pytest.raises(ValueError):
        gun.fired(1.15)
    gun.fired(1.24)
    assert gun.ready_at()==pytest.approx(1.48)


def test_early_timer_wakeup_is_rechecked_before_actual_trigger():
    gun=GunClock(); gun.fired(1.)
    now=[1.1]
    def early_sleep(seconds):
        now[0]+=seconds*.5
    gun.wait_until_ready(clock=lambda:now[0],sleep=early_sleep)
    assert 1.24<=now[0]<1.25
    gun.fired(now[0])


def test_swap_needs_visual_settling_but_not_full_new_reload():
    gun=GunClock()
    gun.fired(1.)
    gun.swapped(1.25)
    assert gun.ready_at()==pytest.approx(1.29)


def test_cyan_guide_is_distinguished_from_a_blue_projectile():
    vision=BoardVision.__new__(BoardVision)
    vision.boards={'test':dict(shooter=np.array([100.,240.]),mask=np.zeros((480,640),np.uint8))}
    image=np.zeros((480,640,3),np.uint8)
    cv2.circle(image,(180,240),16,(20,180,220),-1)
    assert not vision.accuracy_visible(image,'test',[600.,240.])
    cv2.rectangle(image,(145,234),(400,246),(20,180,180),-1)
    assert vision.accuracy_visible(image,'test',[600.,240.])


def test_accuracy_power_reduces_required_motion_lead():
    balls=[Ball(300,0,0,progress=.5)]
    curves=[np.array([[300.,-100.],[300.,0.],[300.,100.]])]
    normal,_,_=shot_geometry(balls,[0.,0.],curves,{0:80},projectile_speed=600)
    fast,_,_=shot_geometry(balls,[0.,0.],curves,{0:80},projectile_speed=1500)
    assert fast[0,1] < normal[0,1]*.6
