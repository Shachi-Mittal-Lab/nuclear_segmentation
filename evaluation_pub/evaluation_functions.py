import numpy as np
import skimage.morphology
import skimage.segmentation
import seaborn as sns

sns.set_style("whitegrid")


def intersection_over_union(ground_truth, prediction):
    """
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
    """

    # Count ground truth and prediction objects
    true_objects = len(np.unique(ground_truth))
    pred_objects = len(np.unique(prediction))

    # Compute 2D histogram containing number of pixels overlapping
    # (intersection) of each pair of ground truth and predicted nuclei
    h = np.histogram2d(
        ground_truth.flatten(), prediction.flatten(), bins=(true_objects,
                                                            pred_objects)
    )
    intersection = h[0]

    # Area of ground truth and predicted objects
    area_true = np.histogram(ground_truth, bins=true_objects)[0]
    area_pred = np.histogram(prediction, bins=pred_objects)[0]

    # Calculating the union number of pixels between each pair of ground truth
    # and predicted nuclei
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)
    union = area_true + area_pred - intersection

    # Exclude background (zero label) from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]

    # Compute Intersection over Union number of pixels between each pair of
    # ground truth and predicted nuclei
    union[union == 0] = 1e-9
    IOU = intersection / union

    return IOU


# Some code inspired from:
# https://github.com/carpenterlab/2019_Caicedo_CytometryA/tree/master
def F1_score_calculator(platform_img, gr_tr_img, threshold, printing=True):
    """
    Calculates F1-score between predicted and
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
        Whether to print details: F1-score and number of true positives, false
        positives, and false negatives

    Returns
    f1: float
        F1-score of nuclear predictions relative to ground truth
    nuclei_count: int
        number of predicted nuclei
    gr_tr_nuclei_count: int
        number of ground truth nuclei
    matches: np.ndarray
        N X M array where N is the number of ground truth nuclei and M is the
        number of predicted nuclei. The array contains boolean values of
        whether each pair of ground truth and predicted nucleus is a true
        positive match. Ground truth nuclei without a match are false negatives
        and predicted nuclei without a match are false positives.
    """

    # Ensures the inputs are binary, black & white arrays
    plat_bin_bool = np.all(np.unique(platform_img.reshape(-1)) == np.array(
        [0, 255]))
    gr_tr_bin_bool = np.all(np.unique(gr_tr_img.reshape(-1)) == np.array(
        [0, 255]))
    assert plat_bin_bool, "Platform mask must be binary black & white."
    assert gr_tr_bin_bool, "Ground truth mask must be binary black & white."

    # Ensures platform and ground truth have the same dimensions
    assert (
        platform_img.shape == gr_tr_img.shape
    ), "Input images do not have the same dimensions"

    # Adding nuclei labels because inputs are binary
    platform_img_labels = skimage.morphology.label(platform_img)
    gr_tr_img_labels = skimage.morphology.label(gr_tr_img)
    gr_tr_img_labels = skimage.segmentation.relabel_sequential(
        gr_tr_img_labels)[0]
    platform_img_labels = skimage.segmentation.relabel_sequential(
        platform_img_labels)[0]

    # number of ground truth nuclei
    gr_tr_nuclei_count = len(np.unique(gr_tr_img_labels)[1:])

    # Calculating IOU matrix for predicted vs ground truth nuclei
    IOU = intersection_over_union(gr_tr_img_labels, platform_img_labels)
    ###
    IOU[IOU < threshold] = 0

    # The below code is to deal with multiple true positives which can arise
    # with IOU thresholds < 0.5
    # Maximum IOU highlighted for each ground truth nucleus
    max_IOU_perGT = np.max(IOU, axis=1)
    
    # The predicted nucleus with the max IOU for each ground truth nucleus
    max_predictionID = np.argmax(IOU, axis=1)
    
    # initializing IOU matrix
    max_IOU_perGT_array = np.zeros_like(IOU)
    
    # Number of ground truth nuclei
    rows = np.arange(IOU.shape[0])
    
    # IOU matrix with only one true positive prediction match per GT
    max_IOU_perGT_array[rows, max_predictionID] = max_IOU_perGT
    
    # The GT nucleus with the max IOU for each prediction
    max_GT_ID = np.argmax(max_IOU_perGT_array, axis=0)
    # Getting the final matches array
    
    nonzero_predictions = np.any(max_IOU_perGT_array != 0, axis=0)
    matches = np.zeros_like(max_IOU_perGT_array, dtype=bool)
    matches[:, nonzero_predictions] = (
        np.arange(max_IOU_perGT_array.shape[0])[:, None]
        == max_GT_ID[nonzero_predictions]
    )
    # end of code to deal with multiple true positives for one GT nucleus

    true_positives = np.sum(matches, axis=1) == 1  # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Extra predicted objects
    false_negatives = np.sum(matches, axis=1) == 0  # Missed true objects

    # Ensures correct accounting
    assert np.all(np.less_equal(true_positives, 1))
    assert np.all(np.less_equal(false_positives, 1))
    assert np.all(np.less_equal(false_negatives, 1))

    # calculates true positives, false positives, and false negatives
    TP, FP, FN = (
        np.sum(true_positives),
        np.sum(false_positives),
        np.sum(false_negatives),
    )
    # Ensures correct accounting
    assert gr_tr_nuclei_count == TP + FN, "Something went wrong with nuclei matching"
    if printing:
        print(f"TP:{TP}, FP:{FP}, FN:{FN}, ground truth nuclei:{gr_tr_nuclei_count}")
    else:
        pass

    # calculates F1 score for a particular platform mask relative to the input
    # ground truth mask
    precision = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN + 1e-9)
    f1 = 2 * ((precision * recall) / (precision + recall + 1e-9))
    jaccard_index = TP / (TP + FP + FN + 1e-9)
    
    # Printing the F1-score if needed
    if printing:
        print(f"F1-score:{f1}")
    else:
        pass

    # number of predicted nuclei
    nuclei_count = TP + FP

    return f1, precision, recall, jaccard_index, nuclei_count, gr_tr_nuclei_count, matches
