from argparse import ArgumentParser
from train_phaser import Phaser
from config import get_config

'''
Example usage:
python3 get_pretrained_model.py --model_key ss-a
'''


# forward function returns tuple of (audio_out, lfo)
def get_pretrained_model(model_key: str, double_precision=True):

    model = Phaser.load_from_checkpoint(get_config(model_key)['ckpt_td'], strict=False).model
    model.eval()
    if double_precision:
        model.double()
    return model


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model_key", type=str, default="ss-a")
    args = parser.parse_args()
    model = get_pretrained_model(model_key=args.model_key)
    print(model)