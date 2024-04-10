import math


import torch
from torch.nn import Parameter
from torch import Tensor as T
import torch.nn.functional as F
from torchaudio.functional import lfilter
import utils as utils
from torchlpc import sample_wise_lpc

class Phaser(torch.nn.Module):
    def __init__(
        self,
        sample_rate,
        window_length=50e-3,
        overlap_factor=0.75,
        mlp_width=16,
        mlp_layers=3,
        mlp_activation="tanh",
        f_range=None,
    ):
        super().__init__()

        ######################
        # Fixed Parameters
        ######################
        self.damped = True  # LFO damping on/off
        self.sample_rate = sample_rate

        ######################
        # Init OLA
        ######################
        self.__init_OLA__(window_length, overlap_factor)

        ######################
        # Learnable Parameters
        ######################
        self.g1 = Parameter(T([1.0]))  # through-path gain
        self.g2 = Parameter(T([0.01]))  # feedback gain
        if f_range is None:  # break-frequency max/min [Hz]
            self.depth = Parameter(0.5 * torch.rand(1))
            self.bias = Parameter(0.1 * torch.rand(1))
        else:
            d_max = T([max(f_range) * 2 / sample_rate])
            d_min = T([min(f_range) * 2 / sample_rate])
            self.depth = Parameter(d_max - d_min)
            self.bias = Parameter(d_min)

        ######################
        # Learnable Modules
        ######################
        self.lfo = utils.DampedOscillator()
        self.mlp = utils.MLP(
            width=mlp_width,
            n_hidden_layers=mlp_layers,
            activation=mlp_activation,
            bias=True,
        )
        self.bq = utils.Biquad(Nfft=self.Nfft, normalise=False)

        ################
        # for logging
        ###############
        self.max_d = 0.0
        self.min_d = 0.0

    def __init_OLA__(self, window_length, overlap_factor):
        self.overlap = overlap_factor
        hops_per_frame = int(1 / (1 - self.overlap))
        self.window_size = hops_per_frame * math.floor(
            window_length * self.sample_rate / hops_per_frame
        )  # ensure constant OLA
        self.hop_size = int(self.window_size / hops_per_frame)
        self.Nfft = 2 ** math.ceil(math.log2(self.window_size) + 1)
        self.register_buffer(
            "window_idx", torch.arange(0, self.window_size, 1).detach()
        )
        self.register_buffer("hann", torch.hann_window(self.window_size).detach())
        self.register_buffer("z", utils.z_inverse(self.Nfft, full=False).view(-1, 1).detach())
        self.OLA_gain = (3 / 8) * (self.window_size / self.hop_size)

    def forward(self, x, sample_based=True):
        device = x.device
        x = x.squeeze()
        sequence_length = x.shape[0]
        num_hops = sequence_length // self.hop_size + 1

        ###########
        # LFO
        ###########
        time = torch.arange(0, num_hops).detach().view(num_hops, 1).to(device)
        lfo = self.lfo(time, damped=self.damped)
        waveshaped_lfo = self.mlp(lfo).squeeze()

        ########################
        # Map to all-pass coeffs
        #######################
        d = self.bias + self.depth * 0.5 * (1 + waveshaped_lfo)
        p = torch.tanh((1.0 - torch.tan(d)) / (1.0 + torch.tan(d)))

        if sample_based:
            return self.forward_sample_based(x, p), p
        else:
            return self.forward_frame_based(x, p), p


    #########################
    # frequency sampling
    ########################
    def forward_frame_based(self, x, p):

        # Filter kernel
        h_ap = torch.pow(((p - self.z) / (1 - p * self.z)), 4)
        h = self.bq().view(-1, 1) * (self.g1 + h_ap / (1 - torch.abs(self.g2) * h_ap))

        X = torch.stft(
            x,
            n_fft=self.Nfft, hop_length=self.hop_size, win_length=self.window_size,
            return_complex=True, onesided=True, center=True, pad_mode="constant", window=self.hann)
        Y = X * h
        y = torch.istft(
            Y,
            n_fft=self.Nfft, win_length=self.window_size, hop_length=self.hop_size,
            window=self.hann, center=True, length=x.shape[-1],
        )

        return y

    #########################
    # time domain
    ########################
    def forward_sample_based(self, x, p):
        sequence_length = x.shape[0]

        # bq filter
        b1 = torch.cat([self.bq.DC, self.bq.ff_params])
        a1 = utils.logits2coeff(self.bq.fb_params)
        h1 = lfilter(x, a1.squeeze(), b1.squeeze(), clamp=False)

        h1g = self.g1 * h1

        allpass_a, allpass_b = utils.fourth_order_ap_coeffs(p)
        combine_b = allpass_b
        combine_a = allpass_a

        combine_denom = combine_a - self.g2.abs() * combine_b
        combine_b = combine_b / combine_denom[..., :1]
        combine_denom = combine_denom / combine_denom[..., :1]

        # upsample
        combine_b = (
            F.interpolate(
                combine_b.T.unsqueeze(0),
                size=sequence_length,
                mode="linear",
                align_corners=True,
            )
            .squeeze(0)
            .T[:sequence_length]
        )
        combine_denom = (
            F.interpolate(
                combine_denom.T.unsqueeze(0),
                size=sequence_length,
                mode="linear",
                align_corners=True,
            )
            .squeeze(0)
            .T[:sequence_length]
        )

        h1h2a = utils.time_varying_fir(h1.unsqueeze(0), combine_b.unsqueeze(0))

        h1h2a = sample_wise_lpc(
            h1h2a, combine_denom[..., 1:].unsqueeze(0)
        ).squeeze()
        return (h1g + h1h2a).unsqueeze(0)


    def get_params(self):
        return {
            "lfo_f0": (self.sample_rate / self.hop_size) * self.lfo.omega / 2 / torch.pi ,
            "lfo_r": self.lfo.get_r(),
            "lfo_phase": self.lfo.phi,
            "dry_mix": self.g1.detach(),
            "feedback": self.g2.detach(),
        }

    def set_frequency(self, f0):
        self.lfo.set_frequency(f0, self.sample_rate / self.hop_size)


