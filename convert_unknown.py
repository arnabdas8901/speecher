import os
import shutil
import sys
sys.path.append("/home/adas/Projects/StarGAN_v2/")
import yaml
import torch
import librosa
import torchaudio
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from munch import Munch
import noisereduce as nr
from Utils.ASR.models import ASRCNN
from Utils.JDC.model import JDCNet
from Speecher.model import Speecher_A2M, Speecher
import soundfile as sf
from parallel_wavegan.utils import load_model
from Speecher.dataset import build_dataloader
from Speecher.train import get_data_path_list
from speechbrain.pretrained import EncoderClassifier


def _load(states, model, force_load=True):
    model_states = model.state_dict()
    for key, val in states.items():
        try:
            if key not in model_states:
                continue
            if isinstance(val, nn.Parameter):
                val = val.data

            if val.shape != model_states[key].shape:
                print(val.shape, model_states[key].shape)
                if not force_load:
                    continue

                min_shape = np.minimum(np.array(val.shape), np.array(model_states[key].shape))
                slices = [slice(0, min_index) for min_index in min_shape]
                model_states[key][slices].copy_(val[slices])
            else:
                model_states[key].copy_(val)
        except:
            print("not exist ", key)

def preprocess(wave):
    to_mel = torchaudio.transforms.MelSpectrogram(
        n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
    mean, std = -4, 4
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

target_speakers = {'5055': 6,
                   '9610': 2,
                   '7194': 4,
                   '8337': 3
                   }

config_path = "/home/adas/Projects/StarGAN_v2/Speecher/Config/config.yml"
config = yaml.safe_load(open(config_path))
vocoder_path = config.get('vocoder_path', None)
device = "cpu"

ASR_config = config.get('ASR_config', False)
ASR_path = config.get('ASR_path', False)
with open(ASR_config) as f:
    ASR_config = yaml.safe_load(f)
ASR_model_config = ASR_config['model_params']
ASR_model = ASRCNN(**ASR_model_config)
params = torch.load(ASR_path, map_location='cpu')['model']
ASR_model.load_state_dict(params)
ASR_model.to(device)

# load pretrained F0 model
F0_path = config.get('F0_path', False)
F0_model = JDCNet(num_class=1, seq_len=192)
params = torch.load(F0_path, map_location='cpu')['net']
F0_model.load_state_dict(params)
F0_model.to(device)

with open("/netscratch/adas/German_vocoder_config.yml") as f:
    voc_config = yaml.load(f, Loader=yaml.Loader)
vocoder = load_model(vocoder_path, config=voc_config)
vocoder.remove_weight_norm()
vocoder.to(device)

model = Speecher_A2M(args=Munch(config['model_params']), asr_model=ASR_model, f0_model=F0_model)
checkpoint_path = "/netscratch/adas/experments/Speecher/speech_generator_onlyMLS_withASR_CB64_PB256_SPK128_both_new/epoch_00074.pth"
state_dict = torch.load(checkpoint_path, map_location="cpu")
_load(state_dict["model_ema"], model)

save_path = "/netscratch/adas/experments/Speecher/Conversion/ood/speech_generator_onlyMLS_withASR_CB64_PB256_SPK128_both_a2m_new/"
os.makedirs(save_path, exist_ok=True)

classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

data = pd.read_csv('/netscratch/adas/experments/VQMIVC/conversion/test_ood/checkpoints-useCSMITrue_useCPMITrue_usePSMITrue_useAmpFalse-500/result.csv')
src_paths = []
ref_paths = []
conv_paths = []
for idx, row in data.iterrows():
    source_path = row['src']
    ref_path = row['ref']
    basename = os.path.basename(source_path).split(".")[0]
    spk_id = os.path.basename(ref_path).split("_")[0]
    spk_code = str(target_speakers[spk_id])
    ref_dataloader = build_dataloader([ref_path+'|' + spk_code + '\n'],
                                         batch_size=1,
                                         num_workers=1,
                                         device= device,
                                         spkr_model=classifier,
                                         validation=True)

    audio, source_sr = librosa.load(source_path, sr=24000)
    if source_sr != 24000:
        audio = librosa.resample(audio, source_sr, 24000)
    #audio = audio / np.max(np.abs(audio))
    audio.dtype = np.float32

    for gen_steps_per_epoch, batch in enumerate(tqdm(ref_dataloader, desc="[generate sample]"), 1):
        batch = [b.to(device) for b in batch]
        _, y_trg, spk_emb = batch
        batch_size = spk_emb.size(0)
        real = preprocess(audio).to(device).unsqueeze(1).repeat(batch_size, 1, 1, 1)
        x_fake = model(real, spk_emb, y_trg, training=False)
        #x_fake = model(real, spk_emb, training=False)
        x_fake = x_fake.transpose(-1, -2)
        y_out = vocoder.inference(x_fake[0].squeeze())
        y_out = y_out.view(-1).cpu().detach().numpy()
        #y_out = np.expand_dims(nr.reduce_noise(y=np.expand_dims(y_out, 0), sr=24000).squeeze(), -1)
        conv_path = os.path.join(save_path, basename+"_converted_"+spk_id+'.wav')
        sf.write(conv_path, y_out.squeeze(), 24000, 'PCM_24')
        src_paths.append(source_path)
        ref_paths.append(ref_path)
        conv_paths.append(conv_path)

df = pd.DataFrame({'src': src_paths,
                   'ref': ref_paths,
                   'converted': conv_paths})
df.to_csv(os.path.join(save_path, 'result.csv'), index=False)
print("End")