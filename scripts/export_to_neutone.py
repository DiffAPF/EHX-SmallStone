import torch as tr
from torch import Tensor
from torch import Tensor as T
from torch import nn
from get_pretrained_model import get_pretrained_model


class PhaserModel(nn.Module):
    def __init__(self, model_key: str, sr: float = 44100) -> None:
        super().__init__()
        self.sr = sr
        self.model = get_pretrained_model(model_key=model_key)
        self.prev_phase = T([0.0])

    def make_argument(self, n_samples: int, freq: float, phase: float) -> T:
        argument = tr.cumsum(2 * tr.pi * tr.full((n_samples,), freq, dtype=tr.double) / self.sr, dim=0) + phase
        return argument

    def forward(self,
                x: Tensor,
                lfo_rate: Tensor,
                lfo_stereo_phase_offset: Tensor) -> Tensor:
        arg_l = self.make_argument(n_samples=x.size(-1), freq=lfo_rate.item(), phase=self.prev_phase.item())
        next_phase = arg_l[-1] % (2 * tr.pi)
        self.prev_phase = next_phase
        arg_r = arg_l + lfo_stereo_phase_offset.item()
        arg = tr.stack([arg_l, arg_r], dim=0)
        lfo = self.model.lfo.amp * tr.cos(arg)
        lfo = lfo.unsqueeze(2)

        d = self.model.bias + self.model.depth * 0.5 * (1 + self.model.mlp(lfo).squeeze())
        p = tr.tanh((1.0 - tr.tan(d)) / (1.0 + tr.tan(d)))

        x = self.model.forward_sample_based(x.repeat(2, 1), p)
        return x