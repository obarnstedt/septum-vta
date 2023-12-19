import cv2
import tqdm
import numpy as np
import scipy.ndimage
from scipy.stats import kde
from matplotlib import pyplot as plt

def get_density(x_y_mean, titles, recording_length, nbins=100, return_values=True):      
    position = x_y_mean
    
    x = position[:,0]
    y = position[:,1]
    
    k = kde.gaussian_kde([x,y])
    xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    
    zi_norm = zi / recording_length
    
    fig = plt.subplots()
    plt.pcolormesh(xi, yi, zi_norm.reshape(xi.shape), cmap="jet")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(False)
    plt.title(titles)
    plt.show()
    
    if return_values:
        return xi, yi, zi


def calculate_background(video):
    
    background_frames = 2000 
    capture = cv2.VideoCapture(video)

    if not capture.isOpened():
        raise Exception("Unable to open video file: {0}".format(video))
        
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame = capture.read()
    
    height, width, _ = frame.shape    
    frames = np.zeros((height,width,background_frames))

    for i in tqdm.tqdm(range(background_frames), disable=not True, desc='Calculate background image for %s' %video):
        rand = np.random.choice(frame_count, replace=False)
        capture.set(1,rand)
        ret, frame = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames[...,i] = gray
    
    print('Finishing up! :)')
    medFrame = np.median(frames,2)
    background = scipy.ndimage.median_filter(medFrame, (5,5))
    
    np.save(video+'-background.npy',background)
    
    capture.release()
    return background


def calculate_approach_time(x_y_mean_up, zone_diameter):
    
    x = x_y_mean_up[:,0]
    y = x_y_mean_up[:,1]
    
    zone_A = []
    zone_B = []
    zone_C = []
    zone_out = []
    for i in range(x_y_mean_up.shape[0]):
        if ((x[i] - 555)**2 + (y[i] - 510)**2 <= zone_diameter[0]**2):
            zone_A.append(x_y_mean_up[i,:])
        
        elif ((x[i] - 555)**2 + (y[i] - 510)**2 <= zone_diameter[1]**2):
            zone_B.append(x_y_mean_up[i,:])
            
        elif ((x[i] - 555)**2 + (y[i] - 510)**2 <= zone_diameter[2]**2):
            zone_C.append(x_y_mean_up[i,:])
           
        else:
            zone_out.append(x_y_mean_up[i,:])
     
    return zone_A, zone_B, zone_C, zone_out


def calculate_approach_pointer(x_y_mean_up, x_y_mean_dn, zone_diameter):
    count_bool = False
    save_axis = True
    
    approach_coord_x = []
    approach_coord_y = []
    x_long = []
    y_long = []
    
    count_approaches_list = []
    for i in range(len(zone_diameter)):
        count_idx = 0
        for num in range(x_y_mean_up.shape[0]):
            x = np.array((x_y_mean_up[num,0], x_y_mean_dn[num,0]))
            y = np.array((x_y_mean_up[num,1], x_y_mean_dn[num,1]))    
            
            coeff = np.polyfit(x, y, 1)
            polynomial = np.poly1d(coeff)
            x_axis = np.linspace(x_y_mean_up[num,0],1000,500)
            y_axis = polynomial(x_axis)
            
            if np.any(((x_axis - 500)**2 + (y_axis - 500)**2 <= zone_diameter[i]**2)): #use 200 diameter to account for inaccuracy of head direction 
                count_bool = True
                if save_axis == True:
                    approach_coord_x.append(x)
                    approach_coord_y.append(y)
                    x_long.append(x_axis)
                    y_long.append(y_axis)
                    save_axis = False
            else:
                if count_bool == True:
                    count_idx += 1
                    count_bool = False
                    save_axis = True
                else:
                    continue
        count_approaches_list.append(count_idx)
        
    return count_approaches_list, approach_coord_x, approach_coord_y


def circle(centers, radius):
    radius = 100
    circle_center_x = 555 
    circle_center_y = 510 
    
    x_coord = []
    for t in range(1000):
        x_temp = circle_center_x + radius * np.cos(0.01*t*2*np.pi)
        x_coord.append(x_temp)
        
    y_coord = []
    for t in range(1000):
        y_temp = circle_center_y + radius * np.sin(0.01*t*2*np.pi)
        y_coord.append(y_temp)
        
    return np.vstack((x_coord,y_coord)).T
