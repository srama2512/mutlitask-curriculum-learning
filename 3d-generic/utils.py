import math
import numpy as np
import pickle
import subprocess
import os

"""
Define the haversine distance function
"""
def haversine_distance(point1, point2):
    """
    Computes the haversine distance (in meters) between 
    two points in the global coordinate systems.
    Source: http://www.movable-type.co.uk/scripts/latlong.html
    point1 - tuple (latitude, longitude) in degrees
    point2 - tuple (latitude, longitude) in degrees
    """
    lat1 = math.radians(point1[0])
    lat2 = math.radians(point2[0])
    delta_lat  = point1[0] - point2[0]
    delta_long = point1[1] - point2[1]

    R = 6371000.0 # In meters
    a = math.sin(math.radians(delta_lat)/2)**2 + math.cos(lat1)*math.cos(lat2)*(math.sin(math.radians(delta_long)/2)**2)
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R*c

    return d

def relative_translation(X1, X2):
    """
    Find relative (x, y, z) translation from X1 to X2 in meters.
    Inputs:
        X1 - tuple(latitude (deg), longitude (deg), height (m))
        X2 - tuple(latitude (deg), longitude (deg), height (m))
    """
    R = 6371000.0 # In meters
    x_ = math.cos(math.radians(X2[0]))*math.sin(math.radians(X2[1]-X1[1]))
    y_ = math.cos(math.radians(X1[0]))*math.sin(math.radians(X2[0])) - \
        math.sin(math.radians(X1[0]))*math.cos(math.radians(X2[0]))*math.cos(math.radians(X2[1]-X1[1]))
    bearing = math.atan2(x_, y_)
    hyp = haversine_distance(X1, X2)
    return (hyp*math.cos(bearing), hyp*math.sin(bearing), X2[2]-X1[2])

def angle_2points(point1, point2):
    """
    point1 - numpy ndarray
    point2 - numpy ndarray
    Ref: https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249
    
    Returns angle between point1 and point2 in degrees
    """
    
    c = np.dot(point1, point2)
    if c == 0:
        return 90.00
    else:
        c /= (np.linalg.norm(point1) * np.linalg.norm(point2))
    return np.degrees(np.arccos(np.clip(c, -1, 1)))

def baseline_angle_1(point1, point2, center):
    """
    point1 - (latitude/phi, longitude/theta, height) triplet
    point2 - (latitude/phi, longitude/theta, height) triplet
    height - (latitude/phi, longitude/theta, height) triplet
    Formula obtained by converting from global coordinates to x,y,z coordinate
    with x, y axes on the equator plane and z from south pole to north pole
    
    Returns baseline angle between viewpoints 1 and 2
    """
    phi1 = math.radians(point1[0])
    theta1 = math.radians(point1[1])
    h1 = point1[2]
    
    phi2 = math.radians(point2[0])
    theta2 = math.radians(point2[1])
    h2 = point2[2]
    
    phic = math.radians(center[0])
    thetac = math.radians(center[1])
    hc = center[2]
    
    R = 6371000.0 # In meters
    
    """
    Convert to x, y, z coordinates
    """
    # Optimization
    cos_phi1 = math.cos(phi1)
    cos_phi2 = math.cos(phi2)
    cos_phic = math.cos(phic)
    R1 = R + h1
    R2 = R + h2
    Rc = R + hc

    x1 = R1 * cos_phi1 * math.sin(theta1)
    y1 = -R1 * cos_phi1 * math.cos(theta1)
    z1 = R1 * math.sin(phi1)
    X1 = np.array([x1, y1, z1])
    
    x2 = R2 * cos_phi2 * math.sin(theta2)
    y2 = -R2 * cos_phi2 * math.cos(theta2)
    z2 = R2 * math.sin(phi2)
    X2 = np.array([x2, y2, z2])
    
    xc = Rc * cos_phic * math.sin(thetac)
    yc = -Rc * cos_phic * math.cos(thetac)
    zc = Rc * math.sin(phic)
    Xc = np.array([xc, yc, zc])
    
    return angle_2points(X1-Xc, X2-Xc)

def baseline_angle_2(point1, point2, center):    
    
    """
    point1 - (latitude/phi, longitude/theta, height) triplet
    point2 - (latitude/phi, longitude/theta, height) triplet
    height - (latitude/phi, longitude/theta, height) triplet
    Formula obtained by computing pairwise haversine distance
    and using cosine formula for triangles
    
    Returns baseline angle between viewpoints 1 and 2
    """
    
    d1 = haversine_distance(point1[:2], center[:2])
    d2 = haversine_distance(point2[:2], center[:2])
    d3 = haversine_distance(point1[:2], point2[:2])
    
    cos_beta = (d1*d1 + d2*d2 - d3*d3)/(2*d1*d2)
    
    return math.degrees(math.acos(cos_beta))

def create_target_cache(dataset_dir, base_dir):
    """
    Given the dataset root directory and the subset name, this function creates
    a target cache which is a dictionary with keys as target IDs and values as 
    a dictionary consisting of the details about the target and its views. The
    cache is saved in the dataset root directory. If the cache is already found,
    it is just loaded.

    Inputs:
        dataset_dir : root dataset directory path
        base_dir    : name of the train data subset like '0002/', '0003/', etc. 

    Outputs:
        targets     : a dictionary with keys as targetIDs. Values are dictionaries
                      containing the following (key, value) pairs:
                        targetCoord: <targetCoord tuple (latitude (degrees), 
                                      longitude (degrees), height (meters))>
                        views      : list of different views of the target. Each 
                                     element of the list contains a dictonary
                                     with 
                                       * 'cameraCoord' as camera coordinates in
                                         (lat, long, ht) 
                                       * 'distance' as distance to
                                        the target point,
                                       * 'imagePath' as path to the view image 
                                         relative to root directory
                                       * 'alignData' as a list of the alignment
                                         values as documented in the dataset

    """
    files = subprocess.check_output(['ls', dataset_dir + base_dir]).split()
    txtfiles = []
    imgfiles = []
    for f in files:
        if f[-3:] == 'txt':
            txtfiles.append(f)
        else:
            imgfiles.append(f)

    print("Number of images read: %d"%(len(txtfiles)))

    """
    Create the dictonary of target points
    """
    targets = {}

    if os.path.isfile(dataset_dir + 'targets_%s.pkl'%(base_dir[:-1])):
        print('Loading saved file')
        targets = pickle.load(open(dataset_dir + 'targets_%s.pkl'%(base_dir[:-1])))
        print('Loaded saved file!')
    else:
        count = 0

        for f in txtfiles:
            strSplit = f.replace('.txt', '').split('_')
            targetID = int(strSplit[3])

            txtPath = base_dir + f
            with open(dataset_dir + txtPath) as infile:
                data = infile.read().split('\n')[:-1]

            if len(data) == 2:
                data = data[0].split()
                targetCoord = (float(data[5]), float(data[6]), float(data[7]))
                targets[targetID] = {'targetCoord': targetCoord, 'views': []}

                count += 1
                #print('Done with %d/%d'%(count, len(txtfiles)))
            #else:
                #print('Ignoring target %d due to no alignment'%(targetID)) 

        count = 0
        for f in txtfiles:
            strSplit = f.replace('.txt', '').split('_')
            targetID = int(strSplit[3])
            datasetID = int(strSplit[0])
            imageID = int(strSplit[1])
            viewID = int(strSplit[2])

            imgPath = base_dir + f.replace('.txt', '.jpg')
            txtPath = base_dir + f
            with open(dataset_dir + txtPath) as infile:
                data = infile.read().split('\n')[:-1]

            if len(data) == 2:
                align_data = data[1].split()
                data = data[0].split()
                targetCoord = map(float, data[5:8])
                cameraCoord = map(float, data[11:14])
                cameraPose = map(float, data[15:18])

                distance = haversine_distance(targetCoord, cameraCoord)
                distance_given = float(data[14])

                #if abs(distance - distance_given) > 0.5:
                    #print('Error in distance computation > 0.5m !')
                    #pdb.set_trace()

                targets[targetID]['views'].append({'cameraCoord': cameraCoord, 'distance': distance_given, 'imagePath': imgPath, \
                        'alignData': align_data, 'cameraPose': cameraPose})
                count += 1
                #print('Done with %d/%d'%(count, len(txtfiles)))

        pickle.dump(targets, open(dataset_dir + 'targets_%s.pkl'%(base_dir[:-1]), 'w'))

    return targets
