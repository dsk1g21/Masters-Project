import numpy as np
#import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plot_file = pd.read_csv(r"C:\masters\machine learning\Provan List\14091996.csv")
df = pd.DataFrame(plot_file)

heatmap_data = df.pivot(index='range_gate', columns='time', values='p_l')
pther_data = df.pivot(index='range_gate', columns='time', values='v')

plt.figure(figsize=(12, 6))
sns.heatmap(pther_data, cmap='gist_rainbow', cbar_kws={'label': 'Velocity (v)'},
            xticklabels=30, yticklabels=10, vmin=-800, vmax=800)  # Adjust tick frequency if needed
#sns.heatmap(heatmap_data, cmap='rainbow', cbar_kws={'label': 'Power (p_l)'}, 
 #           xticklabels=30, yticklabels=10, vmin=0, vmax=40)  # Adjust tick frequency if needed
plt.gca().invert_yaxis()

plt.title('2D Heatmap of Velocity over Time and Range Gate')
plt.xlabel('Time')
plt.ylabel('Range Gate')
plt.xticks(rotation=5)
plt.show()