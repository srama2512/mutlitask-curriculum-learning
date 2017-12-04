import torch
import bottleneck
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

def get_report(predicted_normals, true_normals, mask_normals, num_classes):
    """
    Obtains the binned and unbinned accuracies for surface normal estimation
    Inputs:
        predicted_normals : NxHxW numpy array
        true_normals      : NxHxW numpy array
        mask_normals      : NxHxW numpy array
        num_classes       : total number of classes
    """
    
    mask_normals_reshape = mask_normals.reshape(-1)
    predicted_normals_reshape = predicted_normals.reshape(-1)[mask_normals_reshape > 0]
    true_normals_reshape = true_normals.reshape(-1)[mask_normals_reshape > 0]

    class_counts = []
    sample_weights = np.zeros((predicted_normals_reshape.shape[0]), dtype=np.float32)

    for i in range(num_classes):
        condition = true_normals_reshape == i
        class_counts.append(np.sum(condition))
        sample_weights[condition] = 1.0/float(class_counts[i])

    unbinned_accuracy = accuracy_score(true_normals_reshape, predicted_normals_reshape)
    binned_accuracy = accuracy_score(true_normals_reshape, predicted_normals_reshape, sample_weight=sample_weights)

    return unbinned_accuracy, binned_accuracy

#def get_angular_errors(predicted_probs, true_normals, mask_normals, delaunay_triangles, clusters):
#    """
#    predicted_probs    : N x C x H x W where C is the number of classes. Contains the 
#                         probabilities for each class at every pixel location.
#    true_normals       : N x C x H x W where C = 3. Contains the true normal directions at
#                         each pixel location.
#    mask_normals       : N x H x W numpy array
#    delaunay_triangles : K x 3 numpy array of vertex indices where K is the number of triangles
#    clusters           : 20 x 3 numpy array of normal directions 
#    """
#    partitioned_indices = bottleneck.argpartition(-predicted_probs, 3, axis=1)
#    num_images = true_normals.shape[0]
#    W = true_normals.shape[3]
#    H = true_normals.shape[2]
#
#    for i in range(num_images):

def masked_cross_entropy_2d(logits, targets, masks):
    """
    logits  : N x C x H x W where C is number of classes
    targets : N x H x W
    masks   : N x H x W
    """
    C = logits.size(1)
    logits_flat = logits.permute(0, 2, 3, 1).contiguous().view(-1, C)
    log_probs_flat = F.log_softmax(logits_flat)
    targets_flat = targets.view(-1, 1)
    masks_flat = masks.view(-1, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=targets_flat)
    losses_flat = losses_flat * masks_flat
    loss = loss.es_flat.sum()/masks_flat.sum()
    return loss

