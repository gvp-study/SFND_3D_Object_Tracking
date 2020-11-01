import sys
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./performance.csv')

def analyze_filter(detdes):
    fb = df['Detector+Descriptor']==detdes
    fbd = df[fb]
    fbd
    rows = len(fbd)
    tfbd = np.arange(rows)
    lme = np.mean(fbd['TTC_Lidar'])
    lmd = np.median(fbd['TTC_Lidar'])
    lstd = np.std(fbd['TTC_Lidar'])

    cme = np.mean(fbd['TTC_Camera'])
    cmd = np.median(fbd['TTC_Camera'])
    cstd = np.std(fbd['TTC_Camera'])
    
    print(fbd)
    print('TTC_Lidar: Mean {} Median {} Std {}'.format(lme, lmd, lstd))
    print('TTC_Camera: Mean {} Median {} Std {}'.format(cme, cmd, cstd))

    tit = detdes + "\nLidar Mean " + str(np.round(lme, 2)) + " Std " + str(np.round(lstd, 2))
    tit = tit + ": Camera Mean " + str(np.round(cme, 2)) + " Std " + str(np.round(cstd, 2))

    plt.figure()
    ax = plt.gca()
    fbd.plot(kind='line',x='FrameNumber',y='TTC_Lidar', ax=ax)
    fbd.plot(kind='line',x='FrameNumber',y='TTC_Camera', color='red', grid=True, ax=ax)
    plt.title(tit)
    fname = './results/' + detdes + '.png'
    plt.savefig(fname)

    return np.round(lme, 2), np.round(lmd, 2), np.round(lstd, 2), np.round(cme, 2), np.round(cmd, 2), np.round(cstd, 2) 

if (len(sys.argv) > 1):
    detdes = sys.argv[1];
else:
    detdes = 'FAST+BRIEF'

alldets = df['Detector+Descriptor'].unique()
print(alldets)

plt.rcParams.update({'figure.max_open_warning': 0})

#alldets = alldets[0:2]

with open('./results/means.csv','w') as csvfile:  
    # creating a csv writer object  
    csvwriter = csv.writer(csvfile)  
    fields = ['Detector+Descriptor', 'Lidar_Mean', 'Lidar_Median', 'Lidar_Std', 'Camera_Mean', 'Camera_Median', 'Camera_Std']
    # writing the fields  
    csvwriter.writerow(fields);
  
    for d in alldets:
        [lme, lmd, lstd, cme, cmd, cstd] = analyze_filter(d)
        print(d, lme, lmd, lstd, cme, cmd, cstd)
        csvwriter.writerow([d, lme, lmd, lstd, cme, cmd, cstd])

#analyze_filter('FAST+ORB')

plt.show()

