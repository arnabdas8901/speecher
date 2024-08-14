import os
import sys

import torch

sys.path.append("/home/adas/Projects/StarGAN_v2/")
import warnings
import urllib.request
from wvmos.wv_mos import Wav2Vec2MOS

temp = "/netscratch/adas/Temp/"
path = os.path.join(temp, "wv_mos/wv_mos.ckpt")

if (not os.path.exists(path)):
    print("Downloading the checkpoint for WV-MOS")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    urllib.request.urlretrieve(
        "https://zenodo.org/record/6201162/files/wav2vec2.ckpt?download=1",
        path
    )
    print('Weights downloaded in: {} Size: {}'.format(path, os.path.getsize(path)))


pretrained_mos_model = Wav2Vec2MOS(path, cuda=True)
converted_path = "/netscratch/adas/experments/Speecher/output/samples/speech_generator_onlyMLS_withASR_CB96_PB96_SPK96_FreqAdain_a2m/epoch_60_1__p258_101_target_p8337.wav"
print(converted_path)
warnings.filterwarnings('ignore')
pmos = pretrained_mos_model.calculate_one(converted_path)
print(pmos)