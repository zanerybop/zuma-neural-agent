"""A small learned target ranker. Perception and collision features are explicit.

Bootstrapping uses a documented heuristic teacher, not reward-based self-learning.
Campaign actions are chosen by the trained network, with no teacher fallback.
"""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn

FEATURES = ['color_match', 'run_size', 'progress', 'distance', 'blocked',
            'same_neighbors', 'clearance', 'near_skull', 'confidence', 'chain_size']


class TargetRanker(nn.Module):
    def __init__(self,width=32,output_scale=1.):
        super().__init__()
        self.output_scale=output_scale
        self.network = nn.Sequential(nn.Linear(10,width),nn.ReLU(),nn.Linear(width,width//2),nn.ReLU(),nn.Linear(width//2,1))

    def forward(self, x):
        return self.network(x).squeeze(-1)*self.output_scale


def load_ranker(checkpoint):
    model=TargetRanker(width=int(checkpoint.get('width',32)),output_scale=float(checkpoint.get('output_scale',1.)))
    model.load_state_dict(checkpoint['model'])
    return model.eval()


def teacher_score(features):
    x = features
    immediate_clear = x[...,1] >= .39  # Two existing balls plus the projectile.
    urgency=np.clip(1.5*(1/(1-x[...,2]+.01)-1/.13),0,60)
    return (4*x[...,0] + 8*immediate_clear*x[...,0] + x[...,1]*x[...,0] + .7*x[...,2]
            - .8*x[...,3] - 100*x[...,4] + .3*x[...,5]*x[...,0]
            + .1*x[...,6] + x[...,7]*x[...,0] + .1*x[...,8] + urgency*x[...,0])


def candidates(balls, shooter, color):
    """Construct one shot direction per visible ball and ray obstruction features."""
    if not balls:
        return np.empty((0, 10), np.float32), np.empty((0, 2), np.float32)
    positions = np.array([[b.x, b.y] for b in balls], dtype=float)
    relative = positions-shooter
    distances = np.linalg.norm(relative, axis=1)
    pair_distances = np.linalg.norm(positions[:,None,:]-positions[None,:,:],axis=2)
    tracks = np.array([b.track for b in balls])
    colors = np.array([b.color for b in balls])
    progress = np.array([b.progress for b in balls])
    adjacent = (pair_distances < 42) & (tracks[:,None] == tracks[None,:]) & (np.abs(progress[:,None]-progress[None,:]) < .04)
    np.fill_diagonal(adjacent,False)
    same = adjacent & (colors[:,None] == colors[None,:])
    group_sizes = np.zeros(len(balls),dtype=int)
    visited = set()
    for start in range(len(balls)):
        if start in visited:
            continue
        group, todo = {start}, [start]
        while todo:
            for k in np.flatnonzero(same[todo.pop()]):
                if int(k) not in group:
                    group.add(int(k)); todo.append(int(k))
        visited.update(group)
        group_sizes[list(group)] = len(group)
    features, aims = [], []
    for i, ball in enumerate(balls):
        direction = relative[i]/max(1, distances[i])
        projection = relative@direction
        perpendicular = np.abs(relative[:, 0]*direction[1]-relative[:, 1]*direction[0])
        ahead = (projection > 45) & (projection < distances[i]-20)
        blockers = ahead & (perpendicular < 29)
        clearance = np.min(perpendicular[ahead]) if ahead.any() else 100
        features.append([float(ball.color == color), min(group_sizes[i], 5)/5, ball.progress,
                         distances[i]/800, float(blockers.any()), int(same[i].sum())/2,
                         min(clearance, 100)/100, float(ball.progress > .8),
                         min(ball.confidence, 1), min(len(balls), 100)/100])
        aims.append(positions[i])
    return np.asarray(features, np.float32), np.asarray(aims, np.float32)


def bootstrap(output='runs/ranker/best.pt', real_data=None, seed=42):
    torch.set_num_threads(4); torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x = rng.random((60000, len(FEATURES)), dtype=np.float32)
    x[:, 0] = rng.integers(0, 2, len(x)); x[:, 4] = rng.integers(0, 2, len(x)); x[:, 7] = x[:, 2] > .8
    x[:, 1] = rng.integers(1,6,len(x))/5
    x[:len(x)//2,2]=rng.uniform(.85,1,len(x)//2)
    x[:,7]=x[:,2]>.8
    source = 'synthetic_feature_distillation'
    if real_data:
        arrays = []
        for path in Path(real_data).rglob('candidates-*.npz'):
            with np.load(path) as data:
                arrays.append(data['features'])
        if arrays:
            real = np.concatenate(arrays)
            if len(real) > 60000:
                real = real[rng.choice(len(real), 60000, replace=False)]
            # Include real perception distribution but retain broad feature coverage.
            x = np.concatenate([x, np.tile(real, (max(1, 30000//len(real)), 1))])
            source += '+real_game_candidates'
    y = teacher_score(x).astype(np.float32)
    order = rng.permutation(len(x)); x, y = x[order], y[order]
    split = int(len(x)*.9)
    train_x, train_y = torch.from_numpy(x[:split]), torch.from_numpy(y[:split])
    val_x, val_y = torch.from_numpy(x[split:]), torch.from_numpy(y[split:])
    width,output_scale=64,20.
    model = TargetRanker(width=width,output_scale=output_scale); opt = torch.optim.Adam(model.parameters(), lr=.003)
    for epoch in range(60):
        indices = torch.randperm(len(train_x))
        for batch in indices.split(1024):
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(train_x[batch])/output_scale, train_y[batch]/output_scale); loss.backward(); opt.step()
    model.eval()
    with torch.inference_mode():
        rmse = float(nn.functional.mse_loss(model(val_x), val_y).sqrt())
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(format=1, kind='target_ranker', features=FEATURES, source=source, samples=len(x),
                    width=width,output_scale=output_scale,
                    teacher_version=4, curriculum='immediate_clear_with_nonlinear_terminal_urgency',epochs=60,
                    seed=seed, validation_rmse=rmse, parameters=sum(p.numel() for p in model.parameters()))
    torch.save(dict(**metadata, model=model.state_dict()), output)
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata), flush=True)
    return model
