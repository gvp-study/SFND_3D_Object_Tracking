import sys
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if (len(sys.argv) > 1):
    detdes = sys.argv[1];
else:
    detdes = 'FAST+BRIEF'
    
df = pd.read_csv('./performance.csv')

fb = df['Detector+Descriptor']==detdes
fbd = df[fb]
fbd
rows = len(fbd)
tfbd = np.arange(rows)

print(fbd)
print('TTC_Lidar: Mean {} Median {} Std {}'.format(np.mean(fbd['TTC_Lidar']), np.median(fbd['TTC_Lidar']), np.std(fbd['TTC_Lidar'])))
print('TTC_Camera: Mean {} Median {} Std {}'.format(np.mean(fbd['TTC_Camera']), np.median(fbd['TTC_Camera']), np.std(fbd['TTC_Camera'])))

tit = detdes + "\nLidar Mean " + str(np.round(np.mean(fbd['TTC_Lidar']), 2)) + " Std " + str(np.round(np.std(fbd['TTC_Lidar']), 2))
tit = tit + ": Camera Mean " + str(np.round(np.mean(fbd['TTC_Camera']), 2)) + " Std " + str(np.round(np.std(fbd['TTC_Camera']), 2))

ax = plt.gca()
fbd.plot(kind='line',x='FrameNumber',y='TTC_Lidar', ax=ax)
fbd.plot(kind='line',x='FrameNumber',y='TTC_Camera', color='red', grid=True, ax=ax)
plt.title(tit)
fname = './results/' + detdes + '.png'
plt.savefig(fname)
plt.show()

