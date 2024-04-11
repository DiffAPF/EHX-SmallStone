from typing import Optional

import torch
from torch import Tensor as T
from torch.nn import Parameter
import torch.nn.functional as F


def combine_coeffs(x: T, y: T) -> T:
    pad_size = x.size(-1) - 1
    y = F.pad(y, (pad_size, pad_size)).unfold(-1, x.size(-1), 1)
    return (y @ x.flip(-1).unsqueeze(-1)).squeeze(-1)


def time_varying_fir(x: T, b: T, zi: Optional[T] = None) -> T:
    assert x.ndim == 2
    assert b.ndim == 3
    assert x.size(0) == b.size(0)
    assert x.size(1) == b.size(1)
    order = b.size(2) - 1
    x_padded = F.pad(x, (order, 0))
    if zi is not None:
        assert zi.shape == (x.size(0), order)
        x_padded[:, :order] = zi
    x_unfolded = x_padded.unfold(dimension=1, size=order + 1, step=1)
    x_unfolded = x_unfolded.unsqueeze(3)
    b = b.flip(2).unsqueeze(2)
    y = b @ x_unfolded
    y = y.squeeze(3)
    y = y.squeeze(2)
    return y


def sample_wise_lpc_scriptable(x: T, a: T, zi: Optional[T] = None) -> T:
    assert x.ndim == 2
    assert a.ndim == 3
    assert x.size(0) == a.size(0)
    assert x.size(1) == a.size(1)

    B, T, order = a.shape
    if zi is None:
        zi = a.new_zeros(B, order)
    else:
        assert zi.shape == (B, order)

    padded_y = torch.empty((B, T + order), dtype=x.dtype)
    zi = torch.flip(zi, dims=[1])
    padded_y[:, :order] = zi
    padded_y[:, order:] = x
    a_flip = torch.flip(a, dims=[2])

    for t in range(T):
        padded_y[:, t + order] -= (
            a_flip[:, t : t + 1] @ padded_y[:, t : t + order, None]
        )[:, 0, 0]

    return padded_y[:, order:]


def fourth_order_ap_coeffs(p):
    b = torch.stack([p**4, -4 * p**3, 6 * p**2, -4 * p, torch.ones_like(p)], dim=p.ndim)
    a = b.flip(-1)
    return a, b


def logits2coeff(logits: T) -> T:
    assert logits.shape[-1] == 2
    a1 = torch.tanh(logits[..., 0]) * 2
    a1_abs = torch.abs(a1)
    a2 = 0.5 * ((2 - a1_abs) * torch.tanh(logits[..., 1]) + a1_abs)
    return torch.stack([torch.ones_like(a1), a1, a2], dim=-1)


def z_inverse(num_dft_bins, full=False):
    if full:
        n = torch.arange(0, num_dft_bins, 1)
    else:
        n = torch.arange(0, int(num_dft_bins / 2) + 1, 1)

    omega = 2 * torch.pi * n / num_dft_bins
    real = torch.cos(omega)
    imag = -torch.sin(omega)
    return torch.view_as_complex(torch.stack((real, imag), 1))


class Biquad(torch.nn.Module):
    def __init__(self, Nfft, normalise=False):
        super().__init__()
        self.ff_params = Parameter(T([0.0, 0.0]))
        self.fb_params = Parameter(T([0.0, 0.0]))
        self.DC = Parameter(T([1.0]))
        self.register_buffer("pows", T([1.0, 2.0]))
        self.register_buffer("z", z_inverse(Nfft, full=False).detach().unsqueeze(1))
        self.register_buffer("zpows", torch.pow(self.z, self.pows))
        self.normalise = normalise
        self.Nfft = Nfft

    def forward(self):
        ff = torch.sum(self.ff_params * self.zpows, 1)
        if self.normalise:
            ff += 1.0
        else:
            ff += self.DC
        fb = 1.0 + torch.sum(logits2coeff(self.fb_params).squeeze()[1:] * self.zpows, 1)
        return ff / fb

    def set_Nfft(self, Nfft):
        self.Nfft = Nfft
        self.register_buffer(
            "z", z_inverse(self.Nfft, full=False).detach().unsqueeze(1)
        )
        self.register_buffer("zpows", torch.pow(self.z, self.pows))


class MLP(torch.nn.Module):
    def __init__(
        self,
        in_features=1,
        out_features=1,
        width=8,
        n_hidden_layers=1,
        activation="tanh",
        bias=True,
    ):
        super(MLP, self).__init__()

        self.model = torch.nn.Sequential()

        for n in range(n_hidden_layers):
            self.model.append(
                torch.nn.Linear(in_features=in_features, out_features=width, bias=bias)
            )
            if activation == "tanh":
                self.model.append(torch.nn.Tanh())
            else:
                self.model.append(torch.nn.ReLU())
            in_features = width

        self.model.append(
            torch.nn.Linear(in_features=width, out_features=out_features, bias=bias)
        )

    # requires input shape (L, 1) where L is sequence length
    def forward(self, x):
        y = self.model(x)
        return y.view(x.shape)


class DampedOscillator(torch.nn.Module):

    default_sigma = 0.6
    default_amplitude = 1.0

    def __init__(self):
        super().__init__()

        self.sigma = Parameter(torch.Tensor([self.default_sigma]))
        self.omega = Parameter(0.1 * torch.randn(1))
        self.phi = Parameter(torch.randn(1))
        self.amp = Parameter(torch.Tensor([self.default_amplitude]))

    def forward(self, n: int, damped: bool, normalise: bool = False):

        z = torch.polar(self.get_r(), self.omega)
        z0 = torch.polar(self.amp, self.phi)

        if not damped:
            z = z / torch.abs(z)

        if normalise:
            z0 = z0 / torch.abs(z0)

        return torch.real(z0 * z**n)

    def get_r(self):
        return torch.exp(-self.sigma**2)

    def set_frequency(self, f0, sample_rate):
        omega = 2 * torch.pi * f0 / sample_rate
        self.omega = Parameter(omega)


class ESRLoss(torch.nn.Module):
    def __init__(self):
        super(ESRLoss, self).__init__()
        self.epsilon = 1e-8

    def forward(self, target, predicted):
        mse = torch.mean(torch.square(torch.subtract(target, predicted)))
        signal_energy = torch.mean(torch.square(target))
        return torch.div(mse, signal_energy + self.epsilon)
