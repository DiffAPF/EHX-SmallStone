import os
import pathlib
from typing import List, Dict

import torch as tr
from neutone_sdk import (
    ContinuousNeutoneParameter,
    WaveformToWaveformBase,
    NeutoneParameter,
)
from neutone_sdk.utils import save_neutone_model
from torch import Tensor
from torch import Tensor as T
from torch import nn

from get_pretrained_model import get_pretrained_model


class PhaserModel(nn.Module):
    def __init__(
            self,
            model_key: str,
            sr: int = 44100,
            min_lfo_rate_hz: float = 0.1,
            max_lfo_rate_hz: float = 5.0,
    ) -> None:
        super().__init__()
        self.model_key = model_key
        self.sr = sr
        self.min_lfo_rate_hz = min_lfo_rate_hz
        self.max_lfo_rate_hz = max_lfo_rate_hz
        self.model = get_pretrained_model(model_key=model_key)
        self.model.toggle_scriptable(True)
        self.register_buffer("prev_phase", tr.tensor(0.0, dtype=tr.double))

    def reset(self) -> None:
        self.prev_phase.zero_()

    def make_argument(self, n_samples: int, freq: float, phase: float) -> T:
        argument = (
                tr.cumsum(
                    2 * tr.pi * tr.full((n_samples,), freq, dtype=tr.double) / self.sr,
                    dim=0,
                )
                + phase
        )
        return argument

    def forward(
            self, x: Tensor, lfo_rate_0to1: Tensor, lfo_stereo_phase_offset_0to1: Tensor
    ) -> Tensor:
        assert x.ndim == 2
        n_ch = x.size(0)
        lfo_rate = (
                lfo_rate_0to1 * (self.max_lfo_rate_hz - self.min_lfo_rate_hz)
                + self.min_lfo_rate_hz
        )
        lfo_stereo_phase_offset = lfo_stereo_phase_offset_0to1 * 2 * tr.pi

        arg_l = self.make_argument(
            n_samples=x.size(-1), freq=lfo_rate.item(), phase=self.prev_phase.item()
        )
        next_phase = arg_l[-1] % (2 * tr.pi)
        self.prev_phase = next_phase

        arg_r = arg_l + lfo_stereo_phase_offset
        arg = tr.stack([arg_l, arg_r], dim=0)
        lfo = self.model.lfo.amp * tr.cos(arg)
        lfo = lfo.unsqueeze(2)

        d = self.model.bias + self.model.depth * 0.5 * (
                1.0 + self.model.mlp(lfo).squeeze()
        )
        p = tr.tanh((1.0 - tr.tan(d)) / (1.0 + tr.tan(d)))

        # x = x.repeat(2, 1)
        x = self.model.forward_sample_based(x, p)
        return x


class PhaserModelWrapper(WaveformToWaveformBase):
    def get_model_name(self) -> str:
        return f"phaser_model_{self.model.model_key}"

    def get_model_authors(self) -> List[str]:
        return ["Alistair Carson"]

    def get_model_short_description(self) -> str:
        return "DDSP phaser implementation."

    def get_model_long_description(self) -> str:
        return "DDSP phaser implementation for 'Differentiable All-pole Filters for Time-varying Audio Systems'."

    def get_technical_description(self) -> str:
        return "Wrapper for a DDSP phaser implementation that models an Electro-Harmonix Small Stone phaser pedal."

    def get_technical_links(self) -> Dict[str, str]:
        return {
            # "Paper": "tbd",
            "Code": "https://github.com/DiffAPF/EHX-SmallStone",
        }

    def get_tags(self) -> List[str]:
        return ["phaser", "ddsp", "EHX Small Stone"]

    def get_model_version(self) -> str:
        return "1.0.0"

    def is_experimental(self) -> bool:
        return True

    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        return [
            ContinuousNeutoneParameter(
                "lfo_rate",
                f"LFO rate f[{self.model.min_lfo_rate_hz} Hz to {self.model.max_lfo_rate_hz} Hz]",
                default_value=0.5,
            ),
            ContinuousNeutoneParameter(
                "lfo_stereo_offset",
                f"LFO stereo offset [0 to 2pi]",
                default_value=0.0,
            ),
        ]

    @tr.jit.export
    def is_input_mono(self) -> bool:
        return False

    @tr.jit.export
    def is_output_mono(self) -> bool:
        return False

    @tr.jit.export
    def get_native_sample_rates(self) -> List[int]:
        return [self.model.sr]

    @tr.jit.export
    def get_native_buffer_sizes(self) -> List[int]:
        return []  # Supports all buffer sizes

    @tr.jit.export
    def reset_model(self) -> bool:
        self.model.reset()
        return True

    def do_forward_pass(self, x: T, params: Dict[str, T]) -> T:
        x = x.double()
        lfo_rate_0to1 = params["lfo_rate"]
        lfo_stereo_offset_0to1 = params["lfo_stereo_offset"]
        y = self.model(x, lfo_rate_0to1, lfo_stereo_offset_0to1)
        return y


if __name__ == "__main__":
    model = PhaserModel(model_key="ss-a")
    wrapper = PhaserModelWrapper(model=model)
    root_dir = pathlib.Path(
        os.path.join("neutone_models", wrapper.get_model_name())
    )
    save_neutone_model(
        wrapper,
        root_dir,
        submission=False,
        dump_samples=True,
        test_offline_mode=False,
        speed_benchmark=False,
    )
