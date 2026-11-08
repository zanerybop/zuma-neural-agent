"""Estimate chain speed from successive visible detections, including reversals."""
import numpy as np


class ChainMotion:
    def __init__(self):
        self.previous = None
        self.speeds = {}

    def update(self, name, balls, curves, now):
        if self.previous is None or self.previous[0] != name:
            self.previous = (name, balls, now)
            self.speeds = {}
            return self.speeds
        _, old, before = self.previous
        self.previous = (name, balls, now)
        dt = now-before
        if dt < .025 or dt > .25:
            return self.speeds
        predicted_old = []
        for last in old:
            curve = curves[last.track]
            k = int(last.progress*(len(curve)-1))
            tangent = curve[min(len(curve)-1,k+3)]-curve[max(0,k-3)]
            tangent /= max(1,np.linalg.norm(tangent))
            predicted_old.append(np.array([last.x,last.y])+tangent*self.speeds.get(last.track,0)*dt)
        proposals = []
        for i, current in enumerate(balls):
            for j, last in enumerate(old):
                if current.track == last.track and current.color == last.color:
                    expected = predicted_old[j]
                    distance = np.linalg.norm(np.array([current.x,current.y])-expected)
                    if distance < 32 and np.hypot(current.x-last.x,current.y-last.y) < 90 and abs(current.progress-last.progress) < .06:
                        proposals.append((distance, i, j))
        used_new, used_old, values = set(), set(), {}
        for _, i, j in sorted(proposals):
            if i in used_new or j in used_old:
                continue
            used_new.add(i); used_old.add(j)
            ball, last = balls[i], old[j]
            curve = curves[ball.track]
            nearest = int(ball.progress*(len(curve)-1))
            tangent = curve[min(len(curve)-1, nearest+3)]-curve[max(0, nearest-3)]
            tangent /= max(1, np.linalg.norm(tangent))
            speed = float(np.dot([ball.x-last.x, ball.y-last.y], tangent)/dt)
            if abs(speed) < 200:
                values.setdefault(ball.track, []).append(speed)
        for track, samples in values.items():
            if len(samples) >= 3:
                median = float(np.median(samples))
                self.speeds[track] = .5*self.speeds.get(track, median)+.5*median
        return self.speeds
