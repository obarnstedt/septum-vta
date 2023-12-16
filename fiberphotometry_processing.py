import pandas as pd
import numpy as np
import os
from tqdm.notebook import tqdm, trange
from scipy.stats import zscore
from scipy.signal import hilbert, butter, filtfilt
import sys
from tqdm import tqdm
from igor import binarywave as bw
from statsmodels.nonparametric.smoothers_lowess import lowess

def lickprocessing(licking_raw, SampFreq):
    # licking processing
    LickingZeroed = licking_raw - np.mean(licking_raw)
    # Hilbert-Transform for Envelope:
    Analytic_signal = hilbert(LickingZeroed)
    Licking_Envelope = np.abs(Analytic_signal)
    # Low-Pass Butter filter:
    fc = 2  # Cut-off frequency of the filter
    w = fc / (SampFreq / 2)  # Normalize the frequency to SamplingFrequency
    b, a = butter(5, w, 'low')
    LickingEnvFilt = filtfilt(b, a, Licking_Envelope)
    LickingEnvFilt[0:int(1.5 * SampFreq)] = 0
    licking_filtered = zscore(LickingEnvFilt)

    return licking_filtered

def FPH_processing(fphlp, hz, ds_factor):
    fph_ds = fphlp[::ds_factor]
    polydegree = 2
    x = np.arange(len(fph_ds))
    p = np.polyfit(x, fph_ds, polydegree)
    y = np.polyval(p, x)
    dff = (fph_ds - y) / fph_ds
    # print("Applying LOESS filter on {}x downsampled signal...".format(ds_factor))
    dff_smoothed = lowess(dff, x, (.005*ds_factor)/(hz/1000))[:, 1]
    # print("Finished LOESS filter.")
    zscore = dff_smoothed / np.std(dff_smoothed)
    zscore_upsampled = zscore.repeat(ds_factor)

    return zscore_upsampled

def IGORposition_processing(position_raw, Hz):
    positionrange = int(position_raw.max() - position_raw.min())
    if positionrange < 9:
        positionrange = 9
    lap = pd.Series([1] * len(position_raw))
    IGORpositionjumps = position_raw.index[position_raw.diff() < -4].tolist()
    position_flattened = position_raw.copy()
    for posindex in IGORpositionjumps:
        position_flattened[posindex:] = position_flattened[posindex:] + positionrange

    # KALMAN VELOCITY
    # print("Using Kalman Filter to calculate velocity...")
    posF, velocity = Kalman_Filt_v(position_flattened, 1 / Hz)

    # LAPS
    for idx, posindex in enumerate(IGORpositionjumps):
        lap[posindex:] = lap[posindex:] + 1

    # POSITION SCALING
    BeltMinMax = tuple([0, 360])
    beltscale = abs(BeltMinMax[0]-BeltMinMax[1]) / positionrange
    position = (position_raw+4.5)*beltscale+BeltMinMax[0]

    # VELOCITY CALIBRATION
    total_distance_run_cm = (max(position_flattened) - min(position_flattened)) / positionrange * 360
    total_time = int(len(position_raw)/Hz)
    avg_velocity_real = total_distance_run_cm / total_time
    avg_velocity_measured = np.mean(velocity)
    velocity_calibrated = velocity * (avg_velocity_real / avg_velocity_measured)

    return position.values, velocity_calibrated.values


def Kalman_Filt_v(pos, dt):
    measurements = pos

    #  initialize
    #    x  x'
    x = np.array([[pos[0]], [0]], ndmin=2)  # Initial State (Location and velocity and acceleration)
    P = np.array([[1, 0], [0, 100]])  # Initial Uncertainty
    A = np.array([[1, dt], [0, 1]])  # Transition Matrix

    # Measurement function
    H = np.array([1, 0], ndmin=2)

    # measurement noise covariance; 1e-3 recommended, smaller values for precise onset, larger for smooth velocity
    R = 1e-3

    # Process Noise Covariance
    Q = np.array([[1 / 4 * dt ** 4, 1 / 2 * dt ** 3], [1 / 2 * dt ** 3, dt ** 2]])

    # Identity matrix
    I = np.identity(2)

    #  compare to datapoints
    posF = []
    vF = []

    with tqdm(total=len(measurements)) as pbar:
        for n, measurement in enumerate(measurements):
            # Prediction
            x = np.matmul(A, x)  # predicted State
            P = A @ P @ A.transpose() + Q  # predicted Covariance

            # Correction
            Z = measurement
            y = Z - np.matmul(H, x)  # Innovation from prediction and measurement
            S = H @ P @ H.transpose() + R  # Innovation-covariance
            K = np.matmul(P, H.transpose()) / S  # Filter-Matrix (Kalman-Gain)

            x = x + (K * y)  # recalculation of system state
            # print(x)
            posF.append(np.float64(x[0]))
            vF.append(np.float64(x[1]))

            P = np.matmul(I - (np.matmul(K, H)), P)  # recalculation of covariance
            pbar.update(1)

    return pd.Series(posF), pd.Series(vF)

datafolder = '/Users/Oliver/Google Drive/AG Remy/Petra/MSDBtoVTAfiber/01_Data'

dt = pd.read_excel('/Users/Oliver/Google Drive/AG Remy/Petra/MSDBtoVTAfiber/FS_READ_ME.xlsx', sheet_name='Recordings').rename(
    columns={'Mouse #': 'Mouse'})

hz = 10_000
ds_factor = 1_000

data = []
for recording in trange(len(dt)):
    data_exp = pd.DataFrame()
    dt_exp = dt.loc[recording]
    igorfolder = os.path.join(datafolder, f'Mouse #{dt_exp.Mouse}', dt_exp['Folder name'])
    for (waveno, wave) in zip([0,3,4,6], ['FPH', 'Licking', 'FPH2', 'Position']):
        try:
            raw_signal = bw.load(os.path.join(igorfolder, f'ad{waveno}_{dt_exp.Sweeps}.ibw'))['wave']['wData']
            if wave=='Licking':
                raw_signal = pd.Series(lickprocessing(raw_signal, hz)).abs()
                raw_signal.iloc[:int(1.5*hz)] = np.nan
                raw_signal = raw_signal.interpolate(method='bfill').values
            if 'FPH2' in wave:
                raw_signal = FPH_processing(raw_signal, hz, ds_factor=100)
            if 'Position' in wave:
                raw_signal, velocity = IGORposition_processing(pd.Series(raw_signal), hz)
                data_exp['Velocity'] = np.reshape(velocity, (-1, ds_factor)).mean(axis=1)
                data_exp.loc[(data_exp.Velocity<-5)|(data_exp.Velocity>60), 'Velocity'] = np.nan
                data_exp['Velocity'] = data_exp['Velocity'].interpolate(method='linear')
            data_exp[wave] = np.reshape(raw_signal, (-1, ds_factor)).mean(axis=1)
        except:
            data_exp[wave] = np.nan
    try:
        raw_signal = bw.load(os.path.join(igorfolder, f'ad4_{dt_exp.Sweeps}_LP.ibw'))['wave']['wData']
        raw_signal = FPH_processing(raw_signal, hz, ds_factor=100)
        data_exp['FPH3'] = np.reshape(raw_signal, (-1, ds_factor)).mean(axis=1)
    except:
        data_exp['FPH3'] = np.nan
    data_exp[['Mouse', 'Sweep', 'Date', 'LEDgain', 'Recording']] = dt_exp[['Mouse', 'Sweeps', 'Date', 'LED gain', 'Folder name']]
    data.append(data_exp)
data = pd.concat(data).reset_index().rename(columns={'index': 'sweep_index'})
data.to_hdf(os.path.join(datafolder, 'fph_processed.h5'), 'fph')