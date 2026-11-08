from pathlib import Path
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from .model import TinyPolicy


def load_samples(paths):
    samples = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            x, y, a = (data[k] for k in ('observations', 'targets', 'actions'))
            if x.ndim != 4 or x.shape[1:] != (6, 96, 128) or y.shape != (len(x), 2) or a.shape != (len(x),):
                raise ValueError(f"Invalid sample shapes in {path}")
            if not np.isfinite(y).all() or ((y < 0) | (y > 1)).any() or not np.isin(a, [0, 1, 2]).all():
                raise ValueError(f"Invalid labels in {path}")
            samples.append((x, y, a))
    if not samples:
        raise ValueError('No recorded samples. Run record and press F8 during gameplay.')
    return tuple(torch.from_numpy(np.concatenate([s[i] for s in samples])) for i in range(3))


def split_data(paths):
    sessions = sorted({p.parent for p in paths})
    if len(sessions) >= 2:
        # Entire sessions are held out to avoid almost-identical neighboring frames in validation.
        valid = set(sessions[-max(1, len(sessions)//5):])
        return (load_samples([p for p in paths if p.parent not in valid]),
                load_samples([p for p in paths if p.parent in valid]), 'session')
    data = load_samples(paths)
    split = int(len(data[0])*.8)
    gap = 10
    if split <= gap or len(data[0])-split < 5:
        raise ValueError('Need at least 30 recorded samples; preferably multiple complete sessions.')
    return tuple(t[:split-gap] for t in data), tuple(t[split:] for t in data), 'temporal_with_10_sample_gap'


def train(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    train_data, valid_data, split = split_data(sorted(Path(args.data).rglob('*.npz')))
    if (train_data[2] == 1).sum() < 5:
        raise ValueError('Training data needs at least five shots. Record real gameplay first.')
    counts = torch.bincount(train_data[2].long(), minlength=3).float()
    weights = (counts.sum()/counts.clamp_min(1)).sqrt()
    weights = (weights/weights.mean()).to(device)
    loader = DataLoader(TensorDataset(*train_data), batch_size=args.batch_size, shuffle=True)
    validation = DataLoader(TensorDataset(*valid_data), batch_size=args.batch_size)
    model = TinyPolicy().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    classification = torch.nn.CrossEntropyLoss(weight=weights)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    best = float('inf')
    print(f"Device: {device}; parameters: {sum(p.numel() for p in model.parameters()):,}; train/val: {len(train_data[0])}/{len(valid_data[0])}; split={split}; actions={counts.tolist()}", flush=True)

    def loss_for(batch):
        x, target, actions = [t.to(device) for t in batch]
        aim, logits = model(x)
        mask = actions == 1
        aim_loss = torch.nn.functional.mse_loss(aim[mask], target[mask]) if mask.any() else aim.sum()*0
        return classification(logits, actions.long()) + 5*aim_loss, aim, logits, target, actions

    with (output/'metrics.jsonl').open('w', encoding='utf-8') as metrics:
        for epoch in range(1, args.epochs+1):
            model.train()
            training_loss, n = 0.0, 0
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                loss, *_ = loss_for(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                training_loss += float(loss.detach())*len(batch[0]); n += len(batch[0])
            model.eval()
            val_loss, correct, total, aim_error, shots = 0., 0, 0, 0., 0
            confusion = torch.zeros((3, 3), dtype=torch.int64)
            with torch.inference_mode():
                for batch in validation:
                    loss, aim, logits, target, actions = loss_for(batch)
                    predicted = logits.argmax(1)
                    size = len(actions)
                    val_loss += float(loss)*size; total += size
                    correct += int((predicted == actions).sum())
                    mask = actions == 1
                    aim_error += float(torch.linalg.vector_norm(aim[mask]-target[mask], dim=1).sum())
                    shots += int(mask.sum())
                    confusion += torch.bincount((actions*3+predicted).cpu(), minlength=9).reshape(3, 3)
            result = dict(epoch=epoch, train_loss=training_loss/n, val_loss=val_loss/total,
                          action_accuracy=correct/total, shot_aim_error=aim_error/shots if shots else None,
                          confusion_true_rows=confusion.tolist(), val_shots=shots)
            metrics.write(json.dumps(result)+'\n'); metrics.flush()
            print(json.dumps(result), flush=True)
            if result['val_loss'] < best:
                best = result['val_loss']
                torch.save(dict(format=1, model=model.state_dict(), metrics=result, split=split,
                                train_samples=n, val_samples=total, seed=args.seed), output/'best.pt')
    print(f"Checkpoint: {output/'best.pt'}", flush=True)
