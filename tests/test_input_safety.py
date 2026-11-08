import pytest
from zuma_agent.windows import GameWindow


def test_input_refused_when_game_has_no_focus(monkeypatch):
    window = GameWindow(1, 'test')
    monkeypatch.setattr(window, 'focused', lambda: False)
    monkeypatch.setattr(window, 'region', lambda: pytest.fail('No cursor movement allowed'))
    with pytest.raises(RuntimeError, match='foreground'):
        window.act((.5, .5), 1)


def test_invalid_action_cannot_move_mouse(monkeypatch):
    window = GameWindow(1, 'test')
    monkeypatch.setattr(window, 'focused', lambda: True)
    monkeypatch.setattr(window, 'region', lambda: pytest.fail('No cursor movement allowed'))
    with pytest.raises(ValueError, match='Action'):
        window.act((.5, .5), 3)


def test_game_gets_time_to_aim_before_mouse_down(monkeypatch):
    import zuma_agent.windows as windows
    events = []
    class FakeUser32:
        def SetCursorPos(self,*args):
            events.append('move')
        def mouse_event(self,flags,*args):
            events.append(flags)
    window = GameWindow(1,'test')
    monkeypatch.setattr(window,'focused',lambda:True)
    monkeypatch.setattr(window,'region',lambda:dict(left=0,top=0,width=640,height=480))
    monkeypatch.setattr(windows,'user32',FakeUser32())
    monkeypatch.setattr(windows,'down',lambda key:False)
    monkeypatch.setattr(windows.time,'sleep',lambda seconds:events.append(('wait',seconds)))
    window.act((.5,.5),1)
    assert events == ['move',('wait',.04),0x0002,('wait',.015),0x0004]
