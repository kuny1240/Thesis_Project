import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./Simulate_Result/epoch_reward.csv')
df.plot(y='Min')
plt.show()