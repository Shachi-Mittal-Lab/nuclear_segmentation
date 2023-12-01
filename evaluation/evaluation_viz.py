import numpy as np
import skimage.morphology
import skimage.segmentation
from scipy.spatial.distance import directed_hausdorff as hausdorff
import seaborn as sns
sns.set_style("whitegrid")

def accuracy_calculator(platform_img, gr_tr_img):
    '''
    Calculates pixel-level accuracy of predicted binary nuclear mask relative 
    to ground truth mask

    Parameters
    platform_img: np.ndarray
        Two-dimensional array of platform binary black & white mask
    gr_tr_img: np.ndarray
        Two-dimensional array of ground truth (annotated) binary black & white 
        mask
    '''
    
    # Ensures the inputs are binary, black & white arrays
    plat_bin_bool = np.all(np.unique(platform_img.reshape(-1)) == 
                           np.array([0, 255]))
    gr_tr_bin_bool = np.all(np.unique(gr_tr_img.reshape(-1)) == 
                            np.array([0, 255]))
    
    assert plat_bin_bool, 'Platform mask must be binary black & white.'
    assert gr_tr_bin_bool, 'Ground truth mask must be binary black & white.'
    # Ensures platform and ground truth have the same dimensions
    assert platform_img.shape == gr_tr_img.shape, 'Input images do not have the same dimensions'
    
    # Calculates the accuracy based on the similarity of the pixels between 
    # prediction and ground truth
    num_overlapping_pixels = (platform_img == gr_tr_img).sum()
    accuracy = num_overlapping_pixels / (len(gr_tr_img.reshape(-1)))
    
    return accuracy

def intersection_over_union(ground_truth, prediction):
    '''
    Calculates intersection over union (IoU) of each ground truth and predicted 
    nucleus

    Parameters
    ground_truth: np.ndarray
        Two-dimensional array containing label masks of ground truth nuclei
    prediction: np.ndarray
        Two-dimensional array containing label masks of nuclei predicted by a 
        platform
        
    Returns
    IOU: np.ndarray
        N X M array where N is the number of ground truth nuclei and M is the 
        number of predicted nuclei. The array contains the
        IoU value of each predicted nucleus with a ground truth nucleus
    '''
    
    # Count objects
    true_objects = len(np.unique(ground_truth))
    pred_objects = len(np.unique(prediction))
    
    # Compute intersection
    h = np.histogram2d(ground_truth.flatten(), prediction.flatten(), 
                       bins=(true_objects,pred_objects))
    intersection = h[0]
    
    # Area of objects
    area_true = np.histogram(ground_truth, bins=true_objects)[0]
    area_pred = np.histogram(prediction, bins=pred_objects)[0]
    
    # Calculate union
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)
    union = area_true + area_pred - intersection
    
    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    
    # Compute Intersection over Union
    union[union == 0] = 1e-9
    IOU = intersection/union
    
    return IOU

# Some code inspired from:
# https://github.com/carpenterlab/2019_Caicedo_CytometryA/tree/master
def F1_score_calculator(platform_img, gr_tr_img, threshold, printing=True):
    '''
    Calculates F1-score, average Hausdorff distance between predicted and 
    ground truth nuclei masks based on a IoU threshold

    Parameters
    platform_img: np.ndarray
        Two-dimensional array of platform binary black & white mask
    gr_tr_img: np.ndarray
        Two-dimensional array of ground truth (annotated) binary black & 
        white mask
    threshold: int
        IoU threshold to be considered. Predicted nuclei with lower IoU for 
        a particular ground truth nucleus than the threshold
        will not be considered a true positive. Setting the threshold to be 
        0.5 or higher ensures a maximum of one true positive
        predicted nucleus for every ground truth nucleus
    printing: bool
        Whether to print details
        
    Returns
    f1: float
        F1-score of nuclear predictions relative to ground truth
    aHD: float
        average Hausdorff distance of nuclear predictions relative to 
        ground truth
    nuclei_count: int
        number of predicted nuclei
    gr_tr_nuclei_count: int
        number of ground truth nuclei
    '''
    
    # Ensures the inputs are binary, black & white arrays
    plat_bin_bool = np.all(np.unique(platform_img.reshape(-1)) == 
                           np.array([0, 255]))
    gr_tr_bin_bool = np.all(np.unique(gr_tr_img.reshape(-1)) == 
                            np.array([0, 255]))
    
    assert plat_bin_bool, 'Platform mask must be binary black & white.'
    assert gr_tr_bin_bool, 'Ground truth mask must be binary black & white.'
    # Ensures platform and ground truth have the same dimensions
    assert platform_img.shape == gr_tr_img.shape, 'Input images do not have the same dimensions'
    # Adding nuclei labels because input is a binary image
    platform_img_labels = skimage.morphology.label(platform_img)
    gr_tr_img_labels = skimage.morphology.label(gr_tr_img)
    # number of ground truth nuclei
    gr_tr_nuclei_count = len(np.unique(gr_tr_img_labels)[1:])
    
    gr_tr_img_labels = skimage.segmentation.relabel_sequential(
        gr_tr_img_labels)[0]
    platform_img_labels = skimage.segmentation.relabel_sequential(
        platform_img_labels)[0]
    # Calculating IOU matrix for predicted vs ground truth nuclei
    IOU = intersection_over_union(gr_tr_img_labels, platform_img_labels)
    # Binary nuclei that shows the predicted/ground truth pairs of nuclei with 
    # IOU greater than a preset threshold
    matches = IOU > threshold
    
    true_positives = np.sum(matches, axis=1) == 1   # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Extra objects
    false_negatives = np.sum(matches, axis=1) == 0  # Missed objects
    
    # Ensures correct accounting
    assert np.all(np.less_equal(true_positives, 1))
    assert np.all(np.less_equal(false_positives, 1))
    assert np.all(np.less_equal(false_negatives, 1))
    
    # calculates true positives, false positives, and false negatives
    TP, FP, FN = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    # Ensures correct accounting
    assert gr_tr_nuclei_count == TP+FN, 'Something went wrong with nuclei matching'
    if printing:
        print(f'TP:{TP}, FP:{FP}, FN:{FN}, ground truth nuclei:{gr_tr_nuclei_count}')
    else:
        pass
    # calculates F1 score for a particular platform mask relative to the input 
    # ground truth mask
    f1 = 2*TP / (2*TP + FP + FN + 1e-9)
    # Calculation of average Hausdorff distance (not used here)
    pred_gr_pairs = np.argwhere(matches) + 1
    haus = 0.0
    for i in range(pred_gr_pairs.shape[0]):
        gt_i = np.where(gr_tr_img_labels == pred_gr_pairs[i, 0], 1, 0)
        pred_i = np.where(platform_img_labels == pred_gr_pairs[i, 1], 1, 0)
        seg_ind = np.argwhere(pred_i)
        gt_ind = np.argwhere(gt_i)
        haus += max(hausdorff(seg_ind, gt_ind)[0], 
                    hausdorff(gt_ind, seg_ind)[0])
    
    if pred_gr_pairs.shape[0] != 0:
        aHD = haus/pred_gr_pairs.shape[0]
    else:
        aHD = 0
    if printing:
        print(f'F1-score:{f1}')
    else:
        pass
    # number of predicted nuclei
    nuclei_count = TP + FP
    
    return f1, nuclei_count, gr_tr_nuclei_count, aHD, matches