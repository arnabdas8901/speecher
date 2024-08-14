import numpy as np
import pandas as pd
import scipy.stats as st

csv_path = "/netscratch/adas/experments/Speecher/Conversion/speech_generator_onlyMLS_withASR_CB64_PB256_SPK128_both_a2m_new/result_evaluated.csv"

df = pd.read_csv(csv_path)
pcc_list = df['pcc'].to_list()
mean = np.mean(pcc_list)
print("mean", round(mean, 2))
interval = st.t.interval(confidence=0.95, df=len(pcc_list)-1,
              loc=np.mean(pcc_list),
              scale=st.sem(pcc_list))

print(interval)
print(interval[1]-mean, mean-interval[0])

"""for idx, row in df.iterrows():
    if "/5055/" in row['ref'] or "/7194/" in row['ref']:
        df.loc[idx, 'target'] = "M"
    else:
        df.loc[idx, 'target'] = "F"

    #if "/5295/" in row['src'] or "/2252/" in row['src']:
    if "Eva_K" in row['src']:
        df.loc[idx, 'source'] = "F"
    else:
        df.loc[idx, 'source'] = "M"

df_gen = df.loc[(df['source']== "F") & (df['target']=="F")]

pcc_list = df_gen['cer'].to_list()
mean = np.mean(pcc_list)
print("mean", round(mean, 2))
interval = st.t.interval(confidence=0.95, df=len(pcc_list)-1,
              loc=np.mean(pcc_list),
              scale=st.sem(pcc_list))

print(interval)
print(round((interval[1]-interval[0])/2, 2))"""
