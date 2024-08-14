import os
import sys
sys.path.append("/home/adas/Projects/StarGAN_v2/")
import whisper
import pandas as pd
import urllib.request
from wvmos.wv_mos import Wav2Vec2MOS
from not_to_checkin.cer import cer_whisper
from Utils.evaluation_metrics import mos, pitchCorr_f
from speechbrain.pretrained import SpeakerRecognition

csv_path = "/netscratch/adas/experments/Speecher/Conversion/speech_generator_onlyMLS_withASR_CB64_PB256_SPK128_both_a2m_new/result.csv"
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
pretrained_mos_model = Wav2Vec2MOS(path, cuda=False)

whisper_model = whisper.load_model("medium", download_root = temp)
transcipt_master = open("/ds/audio/mls_german/mls_german/test/transcripts.txt", 'r').readlines()

verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=temp)

df = pd.read_csv(csv_path)
print("CSV loaded")
for idx, row in df.iterrows():
    src = row['src']
    basename = os.path.basename(src).split(".")[0]
    #content = [" ".join(line[:-1].split(" ")[1:]) for line in transcipt_master if basename in line][0]
    #print(content)
    ref = row['ref']
    converted = row['converted']
    pcc = round(pitchCorr_f(src, converted) * 100, 4)
    mos_pred = mos(converted, model=pretrained_mos_model)
    cer, source_text, conv_text = cer_whisper(src, converted, whisper_model, language='de')
    #print(source_text)
    #print(conv_text)

    waveform_x = verification.load_audio(ref)
    waveform_y = verification.load_audio(converted)
    batch_x = waveform_x.unsqueeze(0)
    batch_y = waveform_y.unsqueeze(0)
    score, prediction = verification.verify_batch(batch_x, batch_y, threshold=0.3)

    df.loc[idx, 'pcc'] = pcc
    df.loc[idx, 'mos'] = round(mos_pred, 4)
    df.loc[idx, 'cer'] = round(cer, 4)
    df.loc[idx, 'sim'] = round(score.item(), 4)
    print(idx+1, pcc, mos_pred, cer, score.item())

df.to_csv(os.path.join(os.path.dirname(csv_path), 'result_evaluated.csv'), index=False)