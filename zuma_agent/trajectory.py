"""Predict the first visible ball a shot can hit after applying motion lead."""
import numpy as np


def discard_aim(balls, shooter, margin=45.):
    """Find an unobstructed ray to the screen edge for unusable ammunition."""
    origin=np.asarray(shooter,dtype=float)
    points=np.array([[b.x,b.y] for b in balls],dtype=float).reshape(-1,2)
    best=None
    for angle in np.linspace(0,2*np.pi,360,endpoint=False):
        direction=np.array([np.cos(angle),np.sin(angle)])
        limits=[]
        for k,(low,high) in enumerate(((8,632),(48,472))):
            if abs(direction[k])>1e-6:
                limits.append(((high if direction[k]>0 else low)-origin[k])/direction[k])
        length=min(t for t in limits if t>0)
        target=origin+direction*length
        relative=points-origin
        projection=relative@direction
        perpendicular=np.abs(relative[:,0]*direction[1]-relative[:,1]*direction[0])
        ahead=(projection>35)&(projection<length+32)
        clearance=float(perpendicular[ahead].min()) if ahead.any() else 640.
        if clearance>=margin and (best is None or clearance>best[0]):
            best=(clearance,target.astype(np.float32))
    return best[1] if best else None


def shot_geometry(balls, shooter, curves, speeds, adaptive=True, latency=.025, projectile_speed=600.):
    positions = np.array([[b.x,b.y] for b in balls], dtype=float)
    origin = np.asarray(shooter, dtype=float)
    tangents = []
    for ball in balls:
        curve = curves[ball.track]
        k = int(ball.progress*(len(curve)-1))
        tangent = curve[min(len(curve)-1,k+3)]-curve[max(0,k-3)]
        tangents.append(tangent/max(1,np.linalg.norm(tangent)))
    distances = np.linalg.norm(positions-origin, axis=1)
    leads = np.array([np.clip(speeds.get(b.track,0)*(d/projectile_speed+latency),-120,120)
                      if adaptive else min(14,d/900*40) for b,d in zip(balls,distances)])
    aims = positions+np.array(tangents)*leads[:,None]
    relative = aims-origin
    directions = relative/np.maximum(np.linalg.norm(relative,axis=1,keepdims=True),1)
    projection = directions@relative.T
    perpendicular = np.abs(directions[:,0,None]*relative[None,:,1]-directions[:,1,None]*relative[None,:,0])
    radius = 31.5  # Projectile radius plus chain-ball radius, with a small margin.
    valid = (projection > 50) & (perpendicular < radius)
    collision_distance = projection-np.sqrt(np.maximum(0,radius**2-perpendicular**2))
    collision_distance[~valid] = np.inf
    first_hits = collision_distance.argmin(axis=1)
    first_hits[np.isinf(collision_distance).all(axis=1)] = -1
    return aims.astype(np.float32), first_hits, leads


def choose_feasible(scores, features, balls, first_hits, color):
    eligible = np.flatnonzero(features[:,4] == 0)
    safe = []
    for i in eligible:
        hit = int(first_hits[i])
        if balls[i].color != color or hit < 0 or balls[hit].color != color:
            continue
        same_group = (balls[i].track == balls[hit].track
                      and abs(balls[i].progress-balls[hit].progress) < .04
                      and np.hypot(balls[i].x-balls[hit].x,balls[i].y-balls[hit].y) < 42)
        if hit == i or same_group:
            safe.append(i)
    if safe:
        return int(max(safe,key=lambda i:float(scores[i]))), True
    # If no same-color shot is possible, retain the learned best available shot
    # to cycle ammunition. The controller can swap first if the next color helps.
    choices = eligible if len(eligible) else np.arange(len(scores))
    return int(max(choices,key=lambda i:float(scores[i]))), False
