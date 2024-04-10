from get_pretrained_model import get_pretrained_model
import torchaudio
from config import get_config

'''
Runs all pretrained models and saves the audio output.
'''

if __name__ == '__main__':
    in_sig, sample_rate = torchaudio.load('audio_data/small_stone/input_dry.wav')
    in_sig = in_sig.double()

    keys = ['ss-a', 'ss-b', 'ss-c', 'ss-d', 'ss-e', 'ss-f']
    for k in keys:
        print('Load model from config: {}'.format(k))
        model = get_pretrained_model(model_key=k)
        out_sig, _ = model(in_sig)
        #out_sig, _ = model(in_sig[..., int((60 - get_config(k)['train_data_length']) * sample_rate):])
        torchaudio.save('audio_data/{}.wav'.format(k), out_sig.detach().float(), sample_rate)
