import os
import pandas as pd
from speechbrain.pretrained import SpeakerRecognition

csv_path = "/netscratch/adas/experments/Speecher/Conversion/speech_generator_onlyMLS_withASR_CB64_PB256_SPK128_both_a2m_new/result.csv"
temp = "/netscratch/adas/Temp/"
path = os.path.join(temp, "wv_mos/wv_mos.ckpt")

verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=temp)

df = pd.read_csv(csv_path)
print("CSV loaded")
for idx, row in df.iterrows():
    src = row['src']
    converted = row['converted']

    waveform_x = verification.load_audio(src)
    waveform_y = verification.load_audio(converted)
    batch_x = waveform_x.unsqueeze(0)
    batch_y = waveform_y.unsqueeze(0)
    score, prediction = verification.verify_batch(batch_x, batch_y, threshold=0.3)

    df.loc[idx, 'source_sim'] = round(score.item(), 4)
    print(idx, score)

df.to_csv(os.path.join(os.path.dirname(csv_path), 'result_evaluated_sss.csv'), index=False)