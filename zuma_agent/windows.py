"""Windows client-area capture and input. No game memory access required."""
import ctypes as C
from ctypes import wintypes as W
from dataclasses import dataclass
import os
import time

if os.name != "nt":
    raise RuntimeError("Live game integration requires Windows.")

user32 = C.WinDLL("user32", use_last_error=True)
try:
    user32.SetProcessDpiAwarenessContext(C.c_void_p(-4))
except AttributeError:
    user32.SetProcessDPIAware()

user32.GetForegroundWindow.restype = W.HWND
user32.GetAncestor.argtypes = [W.HWND, W.UINT]
user32.GetAncestor.restype = W.HWND
user32.GetWindowTextW.argtypes = [W.HWND, W.LPWSTR, C.c_int]
user32.GetWindowTextLengthW.argtypes = [W.HWND]
user32.GetClientRect.argtypes = [W.HWND, C.POINTER(W.RECT)]
user32.ClientToScreen.argtypes = [W.HWND, C.POINTER(W.POINT)]
user32.IsWindow.argtypes = [W.HWND]
user32.IsWindowVisible.argtypes = [W.HWND]
user32.IsIconic.argtypes = [W.HWND]
user32.GetAsyncKeyState.argtypes = [C.c_int]
user32.GetAsyncKeyState.restype = C.c_short
CALLBACK = C.WINFUNCTYPE(W.BOOL, W.HWND, W.LPARAM)
user32.EnumWindows.argtypes = [CALLBACK, W.LPARAM]


def windows():
    result = []
    @CALLBACK
    def visit(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            text = C.create_unicode_buffer(user32.GetWindowTextLengthW(hwnd) + 1)
            user32.GetWindowTextW(hwnd, text, len(text))
            if text.value:
                result.append((hwnd, text.value))
        return True
    user32.EnumWindows(visit, 0)
    return result


def down(key):
    return bool(user32.GetAsyncKeyState(key) & 0x8000)


@dataclass
class GameWindow:
    hwnd: int
    title: str

    @classmethod
    def find(cls, title="Zuma Deluxe"):
        matches = [(h, t) for h, t in windows() if title.casefold() in t.casefold()]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one game window containing {title!r}, found {matches}. Run 'launch' or use --title.")
        return cls(*matches[0])

    def region(self):
        if not user32.IsWindow(self.hwnd) or user32.IsIconic(self.hwnd):
            raise RuntimeError("Game window closed or minimized.")
        rect, origin = W.RECT(), W.POINT(0, 0)
        if not user32.GetClientRect(self.hwnd, C.byref(rect)) or not user32.ClientToScreen(self.hwnd, C.byref(origin)):
            raise C.WinError(C.get_last_error())
        if rect.right < 32 or rect.bottom < 32:
            raise RuntimeError("Game client area is too small.")
        return dict(left=origin.x, top=origin.y, width=rect.right, height=rect.bottom)

    def focused(self):
        return user32.GetForegroundWindow() == self.hwnd

    def cursor(self):
        point = W.POINT()
        user32.GetCursorPos(C.byref(point))
        r = self.region()
        return ((point.x-r['left'])/r['width'], (point.y-r['top'])/r['height'])

    def act(self, xy, action):
        if not self.focused():
            raise RuntimeError("Input blocked: Zuma is not the foreground window.")
        if action not in (0, 1, 2):
            raise ValueError("Action must be idle, fire, or swap.")
        r = self.region()
        x, y = [max(0.01, min(0.99, float(v))) for v in xy]
        user32.SetCursorPos(r['left']+round(x*(r['width']-1)), r['top']+round(y*(r['height']-1)))
        if action:
            # Let the game's update loop rotate the frog before delivering MouseDown.
            # A move and click in the same update can fire along the previous angle.
            time.sleep(.04)
            return self.trigger(action)

    def trigger(self, action):
        """Click at the already prepared aim, preserving focus and stop-key guards."""
        if action not in (1, 2):
            raise ValueError('Trigger expects fire or swap')
        if down(0x1B) or down(0x78) or not self.focused():
            return False
        press, release = (0x0002, 0x0004) if action == 1 else (0x0008, 0x0010)
        user32.mouse_event(press, 0, 0, 0, 0)
        try:
            time.sleep(0.015)
        finally:
            user32.mouse_event(release, 0, 0, 0, 0)
        return True
