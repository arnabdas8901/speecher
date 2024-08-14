import pandas as pd
from scipy import stats
base_path = "/netscratch/adas/experments/VQMIVC/conversion/test/checkpoints-useCSMITrue_useCPMITrue_usePSMITrue_useAmpFalse-500/result_evaluated.csv"
df_base = pd.read_csv(base_path)
df_base.reset_index()
df_base.sort_values(by=['src'], inplace=True)
ours_path = "/netscratch/adas/experments/Speecher/Conversion/speech_generator_onlyMLS_withASR_CB64_PB256_SPK128_both_a2m_ema/result_evaluated.csv"
df_ours = pd.read_csv(ours_path)
df_ours.reset_index()
df_ours.sort_values(by=['src'], inplace=True)


res = stats.ttest_rel(df_base['cer'].tolist(), df_ours['cer'].tolist(), nan_policy="omit")
print(res)