from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import numpy as np
import subprocess
import pickle
import math
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

# Source for these two functions
# https://www.learnopencv.com/rotation-matrix-to-euler-angles/
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotationMatrixToEulerAngles(R) :
    assert(isRotationMatrix(R))
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([z, y, x])

def relative_rotation(X1, X2):
    """
    Find relative (yaw, pitch, roll) rotation from X1 to X2 in degrees.
    Inputs:
        X1 - iterable(yaw, pitch, roll)
        X2 - iterable(yaw, pitch, roll)

    Primary references:
    https://en.wikipedia.org/wiki/Rotation_matrix
    https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    """
    yaw1 = math.radians(X1[0])
    pitch1 = math.radians(X1[1])
    roll1 = math.radians(X1[2])
    yaw2 = math.radians(X2[0])
    pitch2 = math.radians(X2[1])
    roll2 = math.radians(X2[2])

    # Use negative of the true angles here
    # NOTE: These rotations are for post multiplication
    rot_z_1 = np.array([[math.cos(yaw1), -math.sin(yaw1), 0], \
                        [math.sin(yaw1), math.cos(yaw1), 0], \
                        [0, 0, 1]])
    rot_y_1 = np.array([[math.cos(pitch1), 0, math.sin(pitch1)], \
                        [0, 1, 0],
                        [-math.sin(pitch1), 0, math.cos(pitch1)]])
    
    # Ignore roll rotation for the purposes of this dataset (roll is always 0)
    rot_z_2 = np.array([[math.cos(yaw2), -math.sin(yaw2), 0], \
                        [math.sin(yaw2), math.cos(yaw2), 0], \
                        [0, 0, 1]])
    rot_y_2 = np.array([[math.cos(pitch2), 0, math.sin(pitch2)], \
                        [0, 1, 0],
                        [-math.sin(pitch2), 0, math.cos(pitch2)]])

    # The transformation is generally rotation about z first, then y. To go from 
    # X1 to X2, we have to undo y first, undo z, then perform z and y for X2.
    rot_undo_1 = np.matmul(rot_y_1.transpose(), rot_z_1.transpose())
    rot_do_2 = np.matmul(rot_z_2, rot_y_2)
    rot_relative = np.matmul(rot_undo_1, rot_do_2)
    
    # Convert rotation matrix to degrees
    angles = rotationMatrixToEulerAngles(rot_relative)

    return [math.degrees(angle_i) for angle_i in angles]

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

def average_angular_error(predicted_angles, true_angles, average=True):
    """
    Angle between predicted pose vector and ground truth vector in the plane defined by their
    cross products. 
    Inputs:
        predicted_angles : Nx3 numpy array
        true_angles      : Nx3 numpy array
        average          : bool (if this is true, it returns average. Otherwise, it returns the
                           error for each element.
    """
    if average:
        avg_error = 0
        for i in range(predicted_angles.shape[0]):
            avg_error +=np.linalg.norm(np.array(relative_rotation(predicted_angles[i, :], true_angles[i, :])))
        
        avg_error /= float(predicted_angles.shape[0])
        
        return float(avg_error)
    else:
        errors = []
        for i in range(predicted_angles.shape[0]):
            errors.append(np.linalg.norm(np.array(relative_rotation(predicted_angles[i, :], true_angles[i, :]))))
        return errors

def average_translation_error(predicted_translations, true_translations, average=True):
    """
    L2 norm of the difference between the normalized translation and ground truth
    vectors. 
    Inputs:
        predicted_translations : Nx3 numpy array
        true_translations      : Nx3 numpy array
        average                : bool (if this is true, it returns average. Otherwise, it returns the
                                 error for each element.

    """
    norm_predicted = np.sqrt(np.sum(predicted_translations * predicted_translations, 1))
    normalized_pred = predicted_translations / np.reshape(norm_predicted, (-1, 1))
    norm_predicted = np.sqrt(np.sum(true_translations * true_translations, 1))
    normalized_true = true_translations / np.reshape(norm_predicted, (-1, 1))
    
    if average:
        avg_error = np.sum((normalized_true - normalized_pred) * (normalized_true - normalized_pred), 1)
        avg_error = np.mean(avg_error)

        return float(avg_error)
    else:
        errors = np.sum((normalized_true - normalized_pred) * (normalized_true - normalized_pred), 1)
        return errors

def auc_score(predicted_probabilities, true_classes, get_roc=False):
    """
    Computes the area under the ROC curve given the binary probabilities
    of predicting class 1 and the true class labels.
    Inputs:
        predicted_probabilities : N numpy array
        true_classes            : N numpy array
        get_roc                 : bool, if True, return also the ROC curve
    """
    if not get_roc:
        return float(roc_auc_score(true_classes, predicted_probabilities))
    else:
        fpr, tpr, thresh = roc_curve(true_classes, predicted_probabilities)
        return float(roc_auc_score(true_classes, predicted_probabilities)), tpr, fpr, thresh
              

