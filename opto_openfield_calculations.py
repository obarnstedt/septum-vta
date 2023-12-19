
import os
from pathlib import Path
import cv2
import tqdm
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
from matplotlib import pyplot as plt
from math import factorial

from opto_openfield_functions import calculate_background, get_density, calculate_approach_time, calculate_approach_pointer
 
def detect_objects_centers(bg, experiment):
    
    if experiment == "OF":
        num_objects = 0
        threshold_lower = 50
        threshold_upper = 255
        object_center = 500
        
    else:
        num_objects = 1
        threshold_lower = 50
        threshold_upper = 255
        object_center = 500
    
    # convert the grayscale image to binary image
    if experiment != "OF":
        ret,thresh = cv2.threshold(bg,threshold_lower,threshold_upper,0) 
        x = np.arange(0,bg.shape[0])
        y = np.arange(0,bg.shape[1])
        
        for i in range(bg.shape[0]):
            for j in range(bg.shape[1]):
                if ((x[i] - object_center)**2 + (y[j] - object_center)**2 >= 150**2): 
                    thresh[i,j] = 0 
                    
        kernel = np.ones((7,7),np.uint8)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
        opening = cv2.dilate(opening,kernel,iterations=9)
        
        opening = opening.astype(np.uint8)
        
        contours, hierarchy = cv2.findContours(opening,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        centroids = np.zeros((num_objects,2))
        for idx, c in enumerate(contours):
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids[idx] = cX, cY
        
        assert len(contours) == num_objects, 'Number of centroids %d differs from number of objects %d' %(len(contours), num_objects)
    
        return centroids, opening

    else:
        return 0, 0

####
#TRAJECTORY CALCULATIONS
###

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0] 

def interpol(arr):
    y = np.transpose(arr) 
    nans, x = nan_helper(y[0])
    y[0][nans]= np.interp(x(nans), x(~nans), y[0][~nans])   
    nans, x = nan_helper(y[1])
    y[1][nans]= np.interp(x(nans), x(~nans), y[1][~nans])
    arr = np.transpose(y)
    
    return arr

def savitzky_golay(y, window_size, order, deriv=0, rate=1):

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    
    return np.convolve( m[::-1], y, mode='valid')
    
def filter_data(x_coords, y_coords, likelihood):
    data_mat=np.column_stack((x_coords, y_coords))
    frame_count=data_mat.shape[0]
    
    # THRESHOLD PARAMETER - 
    accuracy_threshold=0.95

    nan_count=0
    for idx in tqdm.tqdm(range(frame_count), disable=not True, desc='Detect missing values'):
        if likelihood[idx] < accuracy_threshold:
            data_mat[idx,0]=np.nan
            data_mat[idx,1]=np.nan
            nan_count=nan_count+1
   
    print(nan_count)
         
    data_mat = interpol(data_mat)      
    x_Smooth_51 = savitzky_golay(data_mat[:,0], 5, 2) 
    y_Smooth_51 = savitzky_golay(data_mat[:,1], 5, 2)    
    
    return x_Smooth_51, y_Smooth_51

def get_xy_coords(csv):
    """
    Creates an array of all dlc xy-coordinates and a likelihood vector
    
    """
    data = pd.read_csv(csv, skiprows = 1, header=[0,1]) 
    
    data_mat = pd.DataFrame.to_numpy(data)
    data_mat = data_mat[:,1:] 
    
    x_coords=data['nose','x'].to_numpy()
    y_coords=data['nose','y'].to_numpy()
    likelihood=data['nose','likelihood'].to_numpy()
    nose_x_smoothed,nose_y_smoothed = filter_data(x_coords, y_coords, likelihood)
    
    x_coords=data['tail','x'].to_numpy()
    y_coords=data['tail','y'].to_numpy()
    likelihood=data['tail','likelihood'].to_numpy()
    tail_x_smoothed,tail_y_smoothed = filter_data(x_coords, y_coords, likelihood)
    
    x_coords=data['left paw','x'].to_numpy()
    y_coords=data['left paw','y'].to_numpy()
    likelihood=data['left paw','likelihood'].to_numpy()
    rp_x_smoothed,rp_y_smoothed = filter_data(x_coords, y_coords, likelihood)
    
    x_coords=data['right paw','x'].to_numpy()
    y_coords=data['right paw','y'].to_numpy()
    likelihood=data['right paw','likelihood'].to_numpy()
    lp_x_smoothed,lp_y_smoothed = filter_data(x_coords, y_coords, likelihood)
    
    x_coords=data['left hindpaw','x'].to_numpy()
    y_coords=data['left hindpaw','y'].to_numpy()
    likelihood=data['left hindpaw','likelihood'].to_numpy()
    rhp_x_smoothed,rhp_y_smoothed = filter_data(x_coords, y_coords, likelihood)
    
    x_coords=data['right hindpaw','x'].to_numpy()
    y_coords=data['right hindpaw','y'].to_numpy()
    likelihood=data['right hindpaw','likelihood'].to_numpy()
    lhp_x_smoothed,lhp_y_smoothed = filter_data(x_coords, y_coords, likelihood)
    
    
    x_mat_up = np.column_stack((nose_x_smoothed, rp_x_smoothed,lp_x_smoothed))
    y_mat_up = np.column_stack((nose_y_smoothed, rp_y_smoothed,lp_y_smoothed))
    
    x_mat_dn = np.column_stack((tail_x_smoothed, rhp_x_smoothed,lhp_x_smoothed))
    y_mat_dn = np.column_stack((tail_y_smoothed, rhp_y_smoothed,lhp_y_smoothed))
      
    x_mat=np.column_stack((nose_x_smoothed, tail_x_smoothed,rp_x_smoothed,lp_x_smoothed,rhp_x_smoothed,lhp_x_smoothed))
    y_mat=np.column_stack((nose_y_smoothed, tail_y_smoothed,rp_y_smoothed,lp_y_smoothed,rhp_y_smoothed,lhp_y_smoothed))
    
    x_mean_up=np.mean(x_mat_up,axis=1) 
    y_mean_up=np.mean(y_mat_up,axis=1)
    
    x_mean_dn=np.mean(x_mat_dn,axis=1) 
    y_mean_dn=np.mean(y_mat_dn,axis=1)
    
    x_mean=np.mean(x_mat,axis=1)
    y_mean=np.mean(y_mat,axis=1)
    
    x_y_mean_up = np.array((x_mean_up, y_mean_up), order='F').T 
    x_y_mean_dn = np.array((x_mean_dn, y_mean_dn), order='F').T 
    x_y_mean = np.array((x_mean, y_mean), order='F').T 
    
    nose = np.column_stack((nose_x_smoothed,nose_y_smoothed))


    return x_y_mean, x_mean, y_mean, x_y_mean_up, x_y_mean_dn, nose

def get_distance_and_speed(x_y_mean, fps=40, pixel=1065, cm=50): 
    px = cm/pixel 
    dt=1/fps 
    
    dist_list_cm = []
    for i in range(x_y_mean.shape[0]-1):
        x = np.abs(x_y_mean[i+1,0]-x_y_mean[i,0])
        y = np.abs(x_y_mean[i+1,1]-x_y_mean[i,1])
        
        dist_cm = np.sqrt(x ** 2+ y ** 2) * px 
        dist_list_cm.append(dist_cm)

    dist_mat_cm = np.array(dist_list_cm)
    dist_mat_cm = ndimage.median_filter(dist_mat_cm, 9) 
    dist_in_cm = sum(dist_mat_cm)
    
    velo_cm = dist_mat_cm / dt 
    
    velo_downsampled = []
    velo_instant = 0.0
    for i in range(len(velo_cm)):
        velo_instant += velo_cm[i]
        if i % fps == 0:
            velo_instant /= fps
            velo_downsampled.append(velo_instant)
            velo_instant = 0.0
    
    return dist_in_cm, velo_cm, velo_downsampled, dist_mat_cm

def create_stim_signal(recording_length, frame_rate):
    #CREATE STIMULATION SIGNAL
    stim_signal = np.zeros(frame_rate*60)
    stim_start = int(stim_signal.shape[0] / 3)
    stim_end = stim_start * 2
    stim_signal[stim_start:stim_end] = 100
    stim_signal[stim_end:] = 25
    stim_epochs = int(recording_length / stim_signal.shape[0])
    stim_signal = np.tile(stim_signal,stim_epochs)
    
    return stim_signal, stim_epochs

def consecutive(data, stepsize=1):
    data = data[:]
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def trajectory(x_y_mean, stim_signal, bg, file):
    
    pre_idx = np.where(stim_signal == 0)[0]
    stim_idx = np.where(stim_signal == 100)[0]
    post_idx = np.where(stim_signal == 25)[0]
    
    pre_cons = consecutive(pre_idx)
    stim_cons = consecutive(stim_idx)
    post_cons = consecutive(post_idx)
    
    pre_trajectories = np.zeros((10,800,2)) 
    stim_trajectories = np.zeros((10,800,2)) 
    post_trajectories = np.zeros((10,800,2)) 
    
    for i in range(10):
        pre_trajectories[i,...] = x_y_mean[pre_cons[i],:]
        stim_trajectories[i,...] = x_y_mean[stim_cons[i],:]
        post_trajectories[i,...] = x_y_mean[post_cons[i],:]
    
    plt.figure()
    plt.imshow(bg)
    for j in range(9):
        plt.plot(stim_trajectories[j,:,0],stim_trajectories[j,:,1], 'b')
        plt.plot(pre_trajectories[j,:,0],pre_trajectories[j,:,1], 'k')
        plt.plot(post_trajectories[j,:,0],post_trajectories[j,:,1], 'r')
   
    plt.plot(stim_trajectories[9,:,0],stim_trajectories[j,:,1], 'b', label='Stim')
    plt.plot(pre_trajectories[9,:,0],pre_trajectories[j,:,1], 'k', label='Pre')
    plt.plot(post_trajectories[9,:,0],post_trajectories[j,:,1], 'r', label='Post')
    plt.legend()
    plt.title(file)
    
    return stim_trajectories, pre_trajectories, post_trajectories

def main_invivo(main_dir = "main_dir", folder="folder", 
                  condition="obj", frame_rate=40, recording_length=24000, movement_threshold=1.5, 
                  num_objects=1, zone_diameter=[100,150,200]): 
    
    condition="obj"
    frame_rate=40
    recording_length=24000
    movement_threshold=1.5, 
    num_objects=1
    zone_diameter=[100,150,200]
 
    path_to_video = os.path.join(main_dir, folder, "Basler_videos", "")
    path_to_csv = os.path.join(main_dir, folder, "csv", "")
        
    p = Path(path_to_video)
    filenames = [i.stem for i in p.glob('**/*.mp4')]
    print(len(filenames))
    
    overall_df = pd.DataFrame()
    
    #ENTER FILE LOOP INTO SPECIFIED FOLDER 
    for file in filenames:
        print(file)
        #CREATE DICTIONARY FOR VARIABLE OF INTEREST
        data_dict = {"File Name" : [], "Condition" : [], "Opto" : [], "Mean Speed" : [], "SD Speed" : [], "Max Speed" : [], "Distance" : [], 
                     "Time Spent Moving" : [], "Count approaches Zone A" : [], "Count approaches Zone B" : [],
                     "Count approaches Zone C" : [], "Count approaches overall" : [], "Time Approaches Zone A" : [], "Time Approaches Zone B" : [], 
                     "Time Approaches Zone C" : [], "Time Approaches overall" : [], "Time Approaches Out" : []} 
        
        video = os.path.join(path_to_video, file+".mp4")
            
        try:
            bg = np.load(os.path.join(path_to_video, file+'.mp4-background.npy')) #change here 
        except:
            print("No background image found!")
            print("Computing background image ...")
            bg = calculate_background(video)
            
        centroids, opening = detect_objects_centers(bg, condition)
        
        csv = os.path.join(path_to_csv, file+".csv")
        
        x_y_mean, _, _, x_y_mean_up, x_y_mean_dn, nose = get_xy_coords(csv)
        
        titles=file

        xi, yi, zi = get_density(x_y_mean, titles[0], recording_length, nbins=100, return_values=True)
     
        dist_in_cm, velo_cm, velo_downsampled, dist_mat_cm = get_distance_and_speed(x_y_mean, fps=40, pixel=1065, cm=50)
        speed_mean = np.mean(velo_downsampled)
        speed_max = np.max(velo_downsampled)
        speed_std = np.std(velo_downsampled)
        
        time_moving = np.argwhere(velo_cm > movement_threshold)
            
        zone_A, zone_B, zone_C, zone_out = calculate_approach_time(x_y_mean_up, zone_diameter)
        time_zone_a = len(zone_A)/frame_rate
        time_zone_b = len(zone_B)/frame_rate
        time_zone_c = len(zone_C)/frame_rate
        time_zone_out = len(zone_out)/frame_rate
        
        count_idx, approach_coord_x, approach_coord_y = calculate_approach_pointer(x_y_mean_up, x_y_mean_dn, zone_diameter)
        count_overall=(count_idx[0]+count_idx[1]+count_idx[2])
        time_overall=(time_zone_a+time_zone_b+time_zone_c)
    
        data_dict["File Name"].append(file)
        data_dict["Condition"].append(condition)
        data_dict["Opto"].append("Of")
        data_dict["Mean Speed"].append(speed_mean)
        data_dict["SD Speed"].append(speed_std)
        data_dict["Max Speed"].append(speed_max)
        data_dict["Distance"].append(dist_in_cm)
        data_dict["Time Spent Moving"].append(len(time_moving)/frame_rate)
        data_dict["Count approaches Zone A"].append(count_idx[0])
        data_dict["Count approaches Zone B"].append(count_idx[1])
        data_dict["Count approaches Zone C"].append(count_idx[2])
        data_dict["Count approaches overall"].append(count_overall)
        data_dict["Time Approaches Zone A"].append(time_zone_a)
        data_dict["Time Approaches Zone B"].append(time_zone_b)
        data_dict["Time Approaches Zone C"].append(time_zone_c)
        data_dict["Time Approaches overall"].append(time_overall)
        data_dict["Time Approaches Out"].append(time_zone_out)
        
        df=pd.DataFrame(data=data_dict)      
        
        overall_df = overall_df.append(df,ignore_index=True)
        
    return overall_df

Overall_df = main_invivo(main_dir = "main_dir", folder="folder", 
                   condition="object", frame_rate=40, recording_length=24000, movement_threshold=2, 
                   num_objects=0, zone_diameter=[100,150,200])

main_dir = "main_dir"
folder="new_Folder"
Overall_df.to_excel(main_dir + folder + ".xlsx") 
