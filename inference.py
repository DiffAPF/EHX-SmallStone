from get_pretrained_model import get_pretrained_model
import torchaudio

'''
Runs all pretrained models and saves the audio output.
'''

if __name__ == '__main__':
    in_sig, sample_rate = torchaudio.load('audio_data/small_stone/input_dry.wav')

    keys = ['ss-a', 'ss-b', 'ss-c', 'ss-d', 'ss-e', 'ss-f']
    for k in keys:
        print('Load model from config: {}'.format(k))
        model = get_pretrained_model(model_key=k)
        out_sig, _ = model(in_sig.double())
        torchaudio.save('audio_data/{}.wav'.format(k), out_sig.detach().float(), sample_rate)
