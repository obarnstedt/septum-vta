import pandas as pd
import os
import glob
from matplotlib import pyplot as plt
from matplotlib import cm
cmap = cm.get_cmap('Spectral') # Colour map (there are many others)
from tqdm import tqdm
import seaborn as sns
import numpy as np

#define main directory
main_dir = "main_dir"
folder="EPM"
fps = 40

#define csv and video path
path_to_video = os.path.join(main_dir, folder, "Basler_videos", "")
path_to_csv = os.path.join(main_dir, folder, "csv", "")

files = [os.path.basename(f) for f in glob.glob(path_to_csv+'*.csv')]
sorted(files)

#merge all csv files
dlc_raw = []

for filename in tqdm(files):
    dlc = pd.read_csv(os.path.join(path_to_csv, filename))
    bodyparts = list(dlc.iloc[0, 1:].unique()) + ['centroid']
    dlc.columns = [dlc.iloc[0], dlc.iloc[1]]
    dlc = dlc.drop('bodyparts', axis=1).drop([0, 1], axis=0).reset_index(drop=True)
    dlc = dlc.astype(float)
    dlc['Date'] = filename.split('_')[0]
    dlc['Mouse'] = filename[13:17]
    dlc['Opto'] = False if filename[17]=='_' else True
    dlc['Group'] = 'ChR'
    dlc['Experiment'] = filename.split('DLC')[0]
    dlc['Frame'] = dlc.index
    dlc['Time_s'] = dlc.index / fps
    dlc['TimeDiff'] = 1/fps
    dlc.loc[dlc.Mouse.isin(['M1', 'M2', 'M3', 'M4', 'M5', 'M6','M7','M8']), 'Group'] = 'Ctrl'
    dlc.loc[dlc.Mouse.isin(['M1_Arch', 'M2_Arch', 'M3_Arch', 'M4_Arch', 'M5_Arch', 'M6_Arch','M7_Arch']), 'Group'] = 'Arch'
    dlc_raw.append(dlc)
    
dlc_raw = pd.concat(dlc_raw, ignore_index=True)
dlc_raw.groupby('Experiment').max()[['Frame', 'Time_s']]

#define borders of the EPM
xwalls = [480, 555]
ywalls = [475, 555]
cm_px = 70/1000  

# confidence thresholding
conf_thresh = 0.9
dlc_all = dlc_raw.copy().drop('likelihood', axis=1, level=1)
for marker_name in bodyparts[:-1]:
    dlc_all.loc[(dlc_raw.loc[:, (marker_name, 'likelihood')]<conf_thresh), (marker_name, ['x', 'y'])] = np.nan
dlc_all = dlc_all.interpolate(method='linear')

#calculate mouse centroid from DLC file
for x in ['x', 'y']:
    dlc_all['centroid', x] = dlc_all.loc[:,(dlc_all.columns.get_level_values(0).unique(), x)].mean(axis=1)

# smoothed traces
dlc_smoothed = dlc_all.rolling(200, center=True).median()
dlc_smoothed1s = dlc_all.rolling(40, center=True).median()

# Time in centre
dlc_all['InCentre'] = False
dlc_all.loc[(dlc_smoothed.centroid.x>xwalls[0])&(dlc_smoothed.centroid.x<xwalls[1])&(
    dlc_smoothed.centroid.y>ywalls[0])&(dlc_smoothed.centroid.y<ywalls[1]), 'InCentre'] = True

# Distance covered
dlc_all['distance'] = np.linalg.norm(dlc_smoothed['centroid'][['x', 'y']].diff(), axis=1)
dlc_all.loc[dlc_all.Frame==0, 'distance'] = np.nan
dlc_all['distance_cm'] = dlc_all['distance'] * cm_px

# Nose out / Head dipping
dlc_all['NoseOut'] = False
dlc_all.loc[((dlc_smoothed1s.nose.x<xwalls[0]-10)|(dlc_smoothed1s.nose.x>xwalls[1]+10))&((
    dlc_smoothed1s.nose.y<ywalls[0]-10)|(dlc_smoothed1s.nose.y>ywalls[1]+10)), 'NoseOut'] = True
dlc_all['NoseOutOnset'] = dlc_all.NoseOut.astype(int).diff()==1
dlc_all['NoseOutOffset'] = dlc_all.NoseOut.astype(int).diff()==-1
noseouttimes = pd.Series(dlc_all.loc[(dlc_all.NoseOutOnset), 'Time_s'].diff())

# Open arm entries
dlc_all['InOpen'] = False
dlc_all.loc[((dlc_smoothed.centroid.x<xwalls[0])|(dlc_smoothed.centroid.x>xwalls[1])), 'InOpen'] = True
dlc_all['EnterOpen'] = dlc_all.InOpen.astype(int).diff()==1
entrytimes = pd.Series(dlc_all.loc[(dlc_all.EnterOpen), 'Time_s'].diff())

# mouse stats
stats = dlc_all.groupby(['Group', 'Mouse', 'Opto']).mean().reset_index()
stats_max = dlc_all.groupby(['Group', 'Mouse', 'Opto']).max().reset_index()
stats_sum = dlc_all.groupby(['Group', 'Mouse', 'Opto']).sum().reset_index()
seconds = dlc_all.groupby(['Group', 'Mouse', 'Opto']).max()['Time_s'].values
stats_minute = (dlc_all.groupby(['Group', 'Mouse', 'Opto']).sum().apply(lambda x: 60*x/seconds)).reset_index()
stats

#plot EPM trajectories
fig, axs = plt.subplots(len(dlc_all.Mouse.unique()),2, figsize=(4,20), sharex=True, sharey=True)

for j, mouse in enumerate(dlc_all.Mouse.unique()):
    for i, opto in enumerate([False, True]):
        sns.scatterplot(data=dlc_all.loc[(dlc_all.Mouse==mouse)&(dlc_all.Opto==opto), 'centroid'].iloc[::], x='x', y='y',
            c=dlc_all.loc[(dlc_all.Mouse==mouse)&(dlc_all.Opto==opto), 'Time_s'].iloc[::],
            ax=axs[j, i], alpha=0.05, s=1, linewidth=0,
            legend=None).set(xticklabels=[], yticklabels=[], ylabel=None, xlabel=None);
        sns.scatterplot(data=dlc_all.loc[(dlc_all.Mouse==mouse)&(dlc_all.Opto==opto)&(dlc_all.NoseOut), 'centroid'].iloc[::], x='x', y='y',
            color='red', linewidth=0, ax=axs[j, i],
            s=3, legend=None).set(xticklabels=[], yticklabels=[], ylabel=None, xlabel=None);
        sns.scatterplot(data=dlc_all.loc[(dlc_all.Mouse==mouse)&(dlc_all.Opto==opto)&(dlc_all.NoseOutOnset), 'centroid'].iloc[::], x='x', y='y',
            color='black', linewidth=0, ax=axs[j, i],
            s=3, legend=None).set(xticklabels=[], yticklabels=[], ylabel=None, xlabel=None);
        axs[j, i].set_title('_'.join(dlc_all.loc[(dlc_all.Mouse==mouse)&(dlc_all.Opto==opto)][['Group', 'Mouse', 'Opto'
            ]].iloc[0].astype(str).tolist())
            +f" {dlc_all.loc[(dlc_all.Mouse==mouse)&(dlc_all.Opto==opto), 'InOpen'].mean():.2f}", fontsize=8)
        sns.despine(fig=fig, ax=axs[j, i], top=True, right=True, left=True, bottom=True, offset=None, trim=False)
        axs[j, i].tick_params(left=False, bottom=False)

plt.tight_layout()
