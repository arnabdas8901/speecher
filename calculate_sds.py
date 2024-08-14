import numpy as np
import pandas as pd
import scipy.stats as st

csv_path = "/netscratch/adas/experments/Speecher/Conversion/speech_generator_onlyMLS_withASR_CB64_PB256_SPK128_both_a2m_new/result_evaluated_sss.csv"
df = pd.read_csv(csv_path)
df['source_dis'] = 1- df['source_sim']

sds = df['source_dis'].values.tolist()
interval = st.t.interval(confidence=0.95, df=len(sds)-1,
              loc=np.mean(sds),
              scale=st.sem(sds))

print("Mean", np.mean(sds), "CI", interval, "Gap", interval[1]-np.mean(sds), np.mean(sds)-interval[0])