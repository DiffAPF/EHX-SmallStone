<div align="center">
<h1>Differentiable All-pole Filters for Time-varying Audio Systems</h1>

<p>
    <a href="https://yoyololicon.github.io/" target=”_blank”>Chin-Yun Yu</a>,
    <a href="https://christhetr.ee/" target=”_blank”>Christopher Mitcheltree</a>,
    <a href="https://www.linkedin.com/in/alistair-carson-a6178919a/" target=”_blank”>Alistair Carson</a>,
    <a href="https://www.acoustics.ed.ac.uk/group-members/dr-stefan-bilbao/" target=”_blank”>Stefan Bilbao</a>,
    <a href="https://www.eecs.qmul.ac.uk/~josh/" target=”_blank”>Joshua D. Reiss</a>, and
    <a href="https://www.eecs.qmul.ac.uk/~gyorgyf/about.html" target=”_blank”>György Fazekas</a>
</p>

[![Listening Samples](https://img.shields.io/badge/%F0%9F%94%8A%F0%9F%8E%B6-Listening_Samples-blue)](https://diffapf.github.io/web/)
[![Plugins](https://img.shields.io/badge/neutone-Plugins-blue)](https://diffapf.github.io/web/index.html#plugins)
[![License](https://img.shields.io/badge/License-MPL%202.0-orange)](https://www.mozilla.org/en-US/MPL/2.0/FAQ/)

<h2>Phaser (<em>EHX Small Stone</em>) Experiments</h2>
</div>

### Instructions for reproducibility

1) Clone this repo and open its directory.

2) Install requirements using:

    `conda env create -f conda_env_cpu.yml` 
3) Activate environment:

    `conda activate phaser`

### Train models
The recommended training configurations for six snapshots of the EHX Small-Stone pedal considered in the paper can be found in `config.py`.

Train a model with one of these configs using the proposed time-domain filter implementation:\
`python3 train_phaser.py --config ss-a --sample_based`
    
or with the baseline frequency sampleing approach:\
`python3 train_phaser.py --config ss-a `



### Inference
Run inference on all pretrained models provided in the `checkpoints` dir:
```angular2html
python3 inference.py
```
This saves six audio files to  `audio_data/`. 


Note that by default LFO will be out of phase with the equivalent target files in `audio_data/small_stone`. 
This is because the beginning of the training data was truncated to speed up training; only a few cycles of LFO are required to obtain a good model.  To align the phase, comment out line 17 and uncomment line 18 in `inference.py`:
```angular2html
#out_sig, _ = model(in_sig)
out_sig, _ = model(in_sig[..., int((60 - get_config(k)['train_data_length']) * sample_rate):])
```
