"""Visible board perception using installed, read-only game artwork and geometry."""
from dataclasses import dataclass
from pathlib import Path
import os
import struct
import re
import cv2
import numpy as np
from PIL import Image

DEFAULT_GAME = Path(os.environ.get('ZUMA_GAME_DIR', r'C:\Program Files (x86)\Steam\steamapps\common\Zuma Deluxe'))
COLORS = ['blue', 'yellow', 'red', 'green', 'purple', 'white']


def normalize(rgb):
    """Map the centered 4:3 playfield to the game's native 640x480 coordinate space."""
    h, w = rgb.shape[:2]
    scale = min(w/640, h/480)
    cw, ch = round(640*scale), round(480*scale)
    left, top = (w-cw)//2, (h-ch)//2
    return cv2.resize(rgb[top:top+ch, left:left+cw], (640, 480), interpolation=cv2.INTER_AREA)


def to_window(xy, shape):
    h, w = shape[:2]
    scale = min(w/640, h/480)
    return (((w-640*scale)/2 + float(xy[0])*scale)/w,
            ((h-480*scale)/2 + float(xy[1])*scale)/h)


def from_window(xy, shape):
    h, w = shape[:2]
    scale = min(w/640, h/480)
    return np.array([(float(xy[0])*w-(w-640*scale)/2)/scale,
                     (float(xy[1])*h-(h-480*scale)/2)/scale])


def read_curve(path):
    blob = path.read_bytes()
    if blob[:4] != b'CURV':
        raise ValueError(f'Unknown curve format: {path}')
    n = struct.unpack_from('<I', blob, 16)[0]
    if n > 10000 or len(blob) < 20+10*n:
        raise ValueError(f'Invalid curve: {path}')
    points = np.asarray([struct.unpack_from('<ii', blob, 20+10*i) for i in range(n)], dtype=np.float32)
    dense = []
    for a, b in zip(points[:-1], points[1:]):
        dense.extend(a+(b-a)*t for t in np.linspace(0, 1, max(2, int(np.linalg.norm(b-a)/3))))
    return np.asarray(dense, dtype=np.float32)


@dataclass
class Ball:
    x: float
    y: float
    color: int
    radius: float = 16
    progress: float = 0
    track: int = 0
    confidence: float = 1


class BoardVision:
    def __init__(self, root=DEFAULT_GAME):
        self.root = Path(root)
        # PopCap's resource dialect permits unquoted numbers, so it is not strict XML.
        text = (self.root/'levels/levels.xml').read_text(encoding='utf-8')
        text = re.sub(r'<!--.*?-->', '', text, flags=re.S)
        attributes = lambda text: dict(re.findall(r'(\w+)\s*=\s*"([^"]*)"', text))
        self.boards = {}
        self.stages = attributes(re.search(r'<StageProgression\b(.*?)/>', text, re.S).group(1))
        for item in re.findall(r'<Graphics\b([^>]*)>', text):
            meta = attributes(item)
            folder = self.root/'levels'/meta['id']
            image = folder/(meta.get('image', meta['id'])+'.jpg')
            if not image.exists() and meta['id'] != 'space':
                continue
            background = np.asarray(Image.open(image).convert('RGB')) if image.exists() else np.zeros((480, 640, 3), np.uint8)
            curves = []
            for key in ('curve', 'curve2'):
                if meta.get(key) and (folder/(meta[key]+'.dat')).exists():
                    curves.append(read_curve(folder/(meta[key]+'.dat')))
            mask = np.zeros((480, 640), np.uint8)
            for curve in curves:
                cv2.polylines(mask, [curve.astype(np.int32)], False, 255, 52)
            fixed = (mask == 0).astype(np.uint8)
            cv2.circle(fixed, (int(meta['gx']), int(meta['gy'])), 65, 0, -1)
            fixed[:40] = 0
            self.boards[meta['id']] = dict(background=background, small=cv2.resize(background[40:], (80, 55)),
                                          fixed=cv2.resize(fixed[40:], (80, 55), interpolation=cv2.INTER_NEAREST).astype(bool),
                                          curves=curves, mask=mask, shooter=np.array([int(meta['gx']), int(meta['gy'])], dtype=float))
        self.current = None

    def identify(self, image):
        small = cv2.resize(image[40:], (80, 55)).astype(float)
        errors = {name: float(np.mean(np.minimum(np.abs(small-b['small'].astype(float)), 60)[b['fixed']])) for name, b in self.boards.items()}
        # The final nebula scrolls. Its green channel stays dark while purple
        # bands move, so a static RGB template cannot identify that board.
        if 'space' in self.boards:
            errors['space'] = float(np.median(small[:,:,1][self.boards['space']['fixed']])*.5)
        best = min(errors, key=errors.get)
        return best, errors[best]

    def color_masks(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        bright = (s > 115) & (v > 85)
        return [bright & (h >= 85) & (h < 130), bright & (h >= 17) & (h < 36),
                bright & ((h < 17) | (h >= 168)), bright & (h >= 36) & (h < 85),
                bright & (h >= 130) & (h < 168), (s < 80) & (v > 160)]

    def detect(self, image, name=None):
        name = name or self.identify(image)[0]
        board = self.boards[name]
        masks = self.color_masks(image)
        difference = np.max(np.abs(image.astype(np.int16)-board['background'].astype(np.int16)), axis=2) > 35
        if name == 'space':
            # Installed nebula artwork peaks at RGB (105,24,112). Its scrolling
            # purple bands are scenery, not purple balls.
            difference &= image.max(axis=2) > 125
        balls = []
        for color, color_mask in enumerate(masks):
            mask = (color_mask & difference & (board['mask'] > 0)).astype(np.uint8)*255
            mask[:36] = 0
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            peaks = (distance >= cv2.dilate(distance, np.ones((17, 17), np.uint8))-.01) & (distance > 8)
            count, labels, stats, centroids = cv2.connectedComponentsWithStats(peaks.astype(np.uint8))
            for x, y in centroids[1:]:
                if not (18 < x < 622 and 40 < y < 460):
                    continue  # The decorative window frame is not part of the chain.
                if np.linalg.norm(np.array([x, y])-board['shooter']) < 62:
                    continue
                nearest = [(np.min(np.linalg.norm(curve-[x, y], axis=1)), j,
                            np.argmin(np.linalg.norm(curve-[x, y], axis=1))/max(1, len(curve)-1))
                           for j, curve in enumerate(board['curves'])]
                if not nearest:
                    continue
                dist, track, progress = min(nearest)
                if dist < 25:
                    if any(np.linalg.norm(curve[-1]-[x, y]) < 38 for curve in board['curves']):
                        continue
                    balls.append(Ball(float(x), float(y), color, 16, float(progress), track, float(distance[round(y), round(x)]/16)))
        # Color masks and highlights can produce duplicate centers. Prefer the deeper color disk.
        unique = []
        for ball in sorted(balls, key=lambda b: -b.confidence):
            if all(np.hypot(ball.x-b.x, ball.y-b.y) > 20 for b in unique):
                unique.append(ball)
        return sorted(unique, key=lambda b: (b.track, b.progress))

    def shooter_color(self, image, name, aim=None):
        origin = self.boards[name]['shooter']
        masks = self.color_masks(image)
        yy, xx = np.indices(image.shape[:2])
        distance = np.hypot(xx-origin[0], yy-origin[1])
        ring = (distance > 16) & (distance < 37)
        if aim is not None:
            direction = np.asarray(aim)-origin
            direction /= max(1, np.linalg.norm(direction))
            mouth = origin+direction*27
            ring = np.hypot(xx-mouth[0], yy-mouth[1]) < 10
        scores = [np.count_nonzero(m & ring) for m in masks[:5]]
        # White is introduced later; distinguish from the frog's fixed white eyes by
        # requiring a white, background-changed disk in the direction of the mouth.
        winner = int(np.argmax(scores))
        return winner if scores[winner] > (25 if aim is not None else 45) else 5

    def ammo(self, image, name, aim):
        """Read the mouth and rear gem separately, ignoring the frog's blue body."""
        origin = self.boards[name]['shooter']
        direction = np.asarray(aim, dtype=float)-origin
        direction /= max(1, np.linalg.norm(direction))
        def sample(center, radius, minimum):
            x, y = np.round(center).astype(int)
            patch = image[max(0,y-radius):min(480,y+radius+1), max(0,x-radius):min(640,x+radius+1)]
            if patch.shape[:2] != (radius*2+1, radius*2+1):
                return None
            yy, xx = np.indices(patch.shape[:2])
            disk = (xx-radius)**2+(yy-radius)**2 < radius**2
            counts = [np.count_nonzero(m & disk) for m in self.color_masks(patch)]
            hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
            # White Zuma balls are warm ivory, often darker than the empty purple
            # tongue. Brightness alone rejects a genuinely loaded white ball.
            rgb = patch.astype(float)
            counts[5] = np.count_nonzero(disk & (hsv[:,:,1] < 95) & (hsv[:,:,2] > 130)
                                         & (rgb[:,:,0] >= rgb[:,:,2]*.95)
                                         & (rgb[:,:,1] >= rgb[:,:,2]*.90))
            color = int(np.argmax(counts))
            return color if counts[color] >= minimum else None
        return sample(origin+direction*27, 10, 35), sample(origin-direction*24, 5, 12)

    def accuracy_visible(self,image,name,aim):
        """Recognize the cyan aim guide in clear pixels along the current cursor ray."""
        origin=self.boards[name]['shooter']
        direction=np.asarray(aim,dtype=float)-origin
        distance=np.linalg.norm(direction)
        if distance < 105:
            return False
        direction/=distance
        normal=np.array([-direction[1],direction[0]])
        radii=np.arange(45,min(350,distance-5),3)
        points=origin+radii[:,None,None]*direction+np.array([-4,0,4])[None,:,None]*normal
        xy=np.rint(points).astype(int).reshape(-1,2)
        valid=(xy[:,0]>=0)&(xy[:,0]<640)&(xy[:,1]>=32)&(xy[:,1]<480)
        xy=xy[valid]
        if not len(xy):
            return False
        clear=self.boards[name]['mask'][xy[:,1],xy[:,0]]==0
        rgb=image[xy[:,1],xy[:,0]].astype(float)
        cyan=(rgb[:,1]>120)&(rgb[:,2]>120)&(rgb[:,1]>rgb[:,0]*1.4)&(rgb[:,2]>rgb[:,0]*1.4)
        return bool(np.count_nonzero(cyan&clear)>=40 and np.mean(cyan[clear])>.6) if clear.any() else False

    def annotate(self, image, name, balls):
        output = image.copy()
        palette = [(0, 180, 255), (255, 240, 0), (255, 50, 50), (40, 255, 40), (220, 40, 255), (240, 240, 240)]
        for b in balls:
            cv2.circle(output, (round(b.x), round(b.y)), 16, palette[b.color], 2)
            cv2.putText(output, f'{b.color}', (round(b.x)-5, round(b.y)+5), cv2.FONT_HERSHEY_SIMPLEX, .4, (0,0,0), 1)
        cv2.circle(output, tuple(self.boards[name]['shooter'].astype(int)), 42, (255, 255, 255), 1)
        return output
