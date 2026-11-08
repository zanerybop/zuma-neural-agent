from collections import deque
import numpy as np
from PIL import Image

WIDTH, HEIGHT, STACK = 128, 96, 2


class Frames:
    def __init__(self):
        self.frames = deque(maxlen=STACK)

    def push(self, rgb):
        frame = np.asarray(Image.fromarray(rgb).resize((WIDTH, HEIGHT), Image.Resampling.BILINEAR))
        while len(self.frames) < STACK:
            self.frames.append(frame)
        self.frames.append(frame)
        return np.concatenate([f.transpose(2, 0, 1) for f in self.frames], axis=0).copy()


def capture(sct, window):
    return np.asarray(sct.grab(window.region()))[:, :, 2::-1].copy()


def capture_ammo(sct, window, origin, radius=44):
    """Capture only the frog, placed in a native-size canvas for ammo sampling."""
    import cv2
    region = window.region()
    w, h = region['width'], region['height']
    scale = min(w/640, h/480)
    ox, oy = (w-640*scale)/2, (h-480*scale)/2
    x0, y0 = max(0, int(origin[0])-radius), max(0, int(origin[1])-radius)
    x1, y1 = min(640, int(origin[0])+radius+1), min(480, int(origin[1])+radius+1)
    left, top = round(ox+x0*scale), round(oy+y0*scale)
    right, bottom = round(ox+x1*scale), round(oy+y1*scale)
    patch = np.asarray(sct.grab(dict(left=region['left']+left, top=region['top']+top,
                                    width=right-left, height=bottom-top)))[:, :, 2::-1]
    result = np.zeros((480, 640, 3), dtype=np.uint8)
    result[y0:y1, x0:x1] = cv2.resize(patch, (x1-x0, y1-y0), interpolation=cv2.INTER_AREA)
    return result
