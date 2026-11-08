from pathlib import Path
import numpy as np
import pytest
import torch
from zuma_agent.model import TinyPolicy
from zuma_agent.vision import Frames
from zuma_agent.train import load_samples, split_data


def shard(path: Path, n=40):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, observations=np.zeros((n, 6, 96, 128), dtype=np.uint8),
                        targets=np.full((n, 2), .5, dtype=np.float32), actions=np.arange(n)%3)


def test_frames_keep_color_and_history():
    frames = Frames()
    red = np.zeros((96, 128, 3), dtype=np.uint8); red[:, :, 0] = 255
    blue = np.zeros_like(red); blue[:, :, 2] = 255
    frames.push(red)
    stack = frames.push(blue)
    assert stack.shape == (6, 96, 128)
    assert stack[0].min() == 255 and stack[5].min() == 255
    assert stack[2].max() == 0 and stack[3].max() == 0


def test_model_learns_and_checkpoint_roundtrip(tmp_path):
    torch.set_num_threads(2); torch.manual_seed(7)
    model = TinyPolicy()
    x = torch.randint(0, 256, (4, 6, 96, 128), dtype=torch.uint8)
    target = torch.tensor([[.2, .8]]).repeat(4, 1)
    labels = torch.ones(4, dtype=torch.long)
    opt = torch.optim.Adam(model.parameters(), lr=.003)
    def loss():
        xy, logits = model(x)
        return torch.nn.functional.cross_entropy(logits, labels)+5*torch.nn.functional.mse_loss(xy, target)
    initial = float(loss().detach())
    for _ in range(15):
        opt.zero_grad(); error = loss(); error.backward(); opt.step()
    assert float(loss().detach()) < initial*.5
    path = tmp_path/'model.pt'
    torch.save(model.state_dict(), path)
    restored = TinyPolicy(); restored.load_state_dict(torch.load(path, weights_only=True))
    assert torch.allclose(model(x)[0], restored(x)[0])


def test_split_holds_out_whole_session(tmp_path):
    paths = [tmp_path/'session1'/'000.npz', tmp_path/'session2'/'000.npz']
    for p in paths:
        shard(p)
    train, valid, method = split_data(paths)
    assert len(train[0]) == len(valid[0]) == 40
    assert method == 'session'


def test_temporal_split_has_gap(tmp_path):
    path = tmp_path/'one'/'000.npz'; shard(path, n=50)
    train, valid, method = split_data([path])
    assert len(train[0]) == 30 and len(valid[0]) == 10
    assert 'gap' in method


def test_empty_dataset_is_explained():
    with pytest.raises(ValueError, match='No recorded samples'):
        load_samples([])


def test_training_pipeline_writes_loadable_checkpoint(tmp_path):
    from argparse import Namespace
    from zuma_agent.train import train
    for name in ('session1', 'session2'):
        shard(tmp_path/'data'/name/'000.npz')
    output = tmp_path/'run'
    train(Namespace(data=str(tmp_path/'data'), output=str(output), seed=3,
                    device='cpu', batch_size=16, epochs=2, lr=.001))
    checkpoint = torch.load(output/'best.pt', weights_only=True)
    model = TinyPolicy(); model.load_state_dict(checkpoint['model'])
    assert checkpoint['train_samples'] == 40 and checkpoint['val_samples'] == 40
    assert checkpoint['split'] == 'session'
    assert len((output/'metrics.jsonl').read_text().splitlines()) == 2
