import os
import skimage.io as io
import pandas as pd
import numpy as np
import skimage.morphology
import skimage.segmentation
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from evaluation_functions import F1_score_calculator

sns.set_style("whitegrid")


def halfIoUThresh(
        root_dir, regions_coordinate_dict, platforms_list,
        save_qual_imgs=False, print_qual_imgs=False):
    """
    Produces csv files containing F1-scores for quantitative evaluation.
    Optionally produces visualizations of predicted nuclei overlaid on ground
    truth nuclei for qualitative evaluation.
    Both of these are performed at an intersection over union threshold of 0.5
    for F1-score calculation.

    Parameters
    root_dir: str
        the file path to the root directory. See the README.md file for details
        about the required directory structure to perform the evaluation.
    region_coordinate_dict: dict
        dictionary containing the coordinates of the top-left pixel of each
        evaluation subregion from each whole slide image. See evaluation.ipynb
        notebook for an example.
    platform_list: list
        list of strings containing the name of the segmentation platforms being
        evaluated
    save_qual_imgs: bool
        whether to save the visualizations of predicted nuclei overlaid on
        ground truth nuclei for qualitative evaluation
    print_qual_imgs: bool
        whether to print the visualizations of predicted nuclei overlaid on
        ground truth nuclei for qualitative evaluation
    """

    # creating a list of WSI fields
    regions_dir_path = os.path.join(root_dir, "Fields")
    regions_dir_files = os.listdir(regions_dir_path)
    regions_list = []
    for file in regions_dir_files:
        file_path = os.path.join(regions_dir_path, file)
        if os.path.isdir(file_path):
            regions_list.append(file)
        else:
            pass

    # iterating over all WSI fields
    for region in regions_list:
        # getting file paths for platforms and ground truth binary masks
        platforms_dir_path = os.path.join(regions_dir_path, region,
                                          "Platforms")
        gr_tr_dir_path = os.path.join(regions_dir_path, region, "ground_truth")
        # getting the coordinates of the evaluation subregions for this region
        coordinate_dict = regions_coordinate_dict[region]

        # making the output directory if it does not exist
        if not os.path.exists(os.path.join(root_dir, "F1_halfIoU_csv")):
            os.makedirs(os.path.join(root_dir, "F1_halfIoU_csv"))
        else:
            pass
        F1_halfIoU_csv_path = os.path.join(root_dir, "F1_halfIoU_csv")

        # reading the ground truth binary masks
        gr_tr_sparse1 = io.imread(os.path.join(gr_tr_dir_path,
                                               "gr_tr_sparse1.tiff"))
        gr_tr_sparse2 = io.imread(os.path.join(gr_tr_dir_path,
                                               "gr_tr_sparse2.tiff"))
        gr_tr_dense1 = io.imread(os.path.join(gr_tr_dir_path,
                                              "gr_tr_dense1.tiff"))
        gr_tr_dense2 = io.imread(os.path.join(gr_tr_dir_path,
                                              "gr_tr_dense2.tiff"))

        # creating a dictionary with the ground truth binary masks
        gr_tr_mask_dict = {
            "sparse1": gr_tr_sparse1,
            "sparse2": gr_tr_sparse2,
            "dense1": gr_tr_dense1,
            "dense2": gr_tr_dense2,
        }

        # this dictionary will be populated with subfield names and their
        # associated F1 scores for each platform
        df_dict_F1 = {"subfield": []}
        for i in platforms_list:
            df_dict_F1[i] = []

        # initializing the ground truth nuclei count to be incremented
        gr_tr_nuclei_count = []
        # iterating over all 4 subfields
        for subfield in gr_tr_mask_dict.keys():
            # iterating over each platform's binary mask
            for img in os.listdir(platforms_dir_path):
                # processes only tiff files
                if img[-5:] == ".tiff" or img[-4:] == ".tif":
                    platform_whole_img = io.imread(
                        os.path.join(platforms_dir_path, img)
                    )
                    platform = img[: img.find(".")]
                    # adds a new entry in the dictionary for the subfield
                    # if it is not already added
                    if subfield not in df_dict_F1["subfield"]:
                        df_dict_F1["subfield"].append(subfield)
                    else:
                        pass
                    # gets the ground truth mask for the current subfield
                    gr_tr_img = gr_tr_mask_dict[subfield]
                    # gets the coordinates for the current subfield
                    top_left_row = coordinate_dict[subfield][0]
                    top_left_column = coordinate_dict[subfield][1]
                    # crops the platform binary mask to isolate the region to
                    # be evaluated
                    # the evaluation subregion size is 256 X 256 pixels
                    platform_img = platform_whole_img[
                        top_left_row: top_left_row + 256,
                        top_left_column: top_left_column + 256,
                    ]
                    # Not considering nuclei touching the region borders
                    platform_img = skimage.segmentation.clear_border(
                        platform_img)
                    gr_tr_img = skimage.segmentation.clear_border(gr_tr_img)
                    # creating a ground truth binary mask for visualization
                    # purposes in white and black
                    gr_tr_img_show = np.uint8(
                        np.zeros((gr_tr_img.shape[0], gr_tr_img.shape[1], 3))
                    )
                    gr_tr_img_show[gr_tr_img == 255] = [255, 255, 255]
                    # creating a ground truth binary mask for visualization
                    # purposes in green and black
                    platform_img_show = np.uint8(
                        np.zeros((platform_img.shape[0], platform_img.shape[1],
                                  3))
                    )
                    platform_img_show[platform_img == 255] = [255, 99, 71]
                    # overlaying the platform over the ground truth binary mask
                    dst = cv2.addWeighted(
                        gr_tr_img_show, 0.6, platform_img_show, 0.9, 0
                    )
                    dst[np.all(dst == (230, 89, 64), axis=-1)] = (255, 99, 71)
                    dst[np.all(dst == (255, 242, 217), axis=-1)] = (
                        255, 203, 203)

                    # for visualization in the notebook
                    if print_qual_imgs:
                        plt.imshow(dst)
                        plt.show()
                    else:
                        pass
                    # to save the overlayed images for qualitative evaluation
                    if save_qual_imgs:
                        if not os.path.exists(
                            os.path.join(root_dir, "qual_imgs", region,
                                         "halfIoU")
                        ):
                            os.makedirs(
                                os.path.join(root_dir, "qual_imgs", region,
                                             "halfIoU")
                            )
                        else:
                            pass
                        eval_images_halfIoU_path = os.path.join(
                            os.path.join(root_dir, "qual_imgs", region,
                                         "halfIoU")
                        )
                        io.imsave(
                            os.path.join(
                                eval_images_halfIoU_path,
                                platform + "_" + subfield + ".tiff",
                            ),
                            dst,
                            check_contrast=False,
                        )
                    else:
                        pass

                    # calculating the F1-score of the current platform subfield
                    (
                        F1,
                        nuclei_count,
                        cur_gr_tr_nuclei_count,
                        aHD,
                        matches,
                    ) = F1_score_calculator(
                        platform_img, gr_tr_img, 0.5, printing=print_qual_imgs
                    )

                    # printing the qualitative images if needed
                    if print_qual_imgs:
                        print(platform + "_" + subfield)
                        print("----------------------------------------------")
                    else:
                        pass
                    # Adding the F1-score to the F1-score dictionary
                    df_dict_F1[platform].append(F1)
                else:
                    pass
            # updating the ground truth nuclei count for the current subfield
            gr_tr_nuclei_count.append(cur_gr_tr_nuclei_count)
        # Adding up the ground truth nuclei in all the subfields
        gr_tr_nuclei_count = sum(gr_tr_nuclei_count)

        # Converting the F1-score dictionary to a dataframe
        df_F1 = pd.DataFrame(df_dict_F1)
        # Setting the index to be the subfield name
        df_F1 = df_F1.set_index("subfield")
        # converting the pandas dataframe to a csv file and saving it
        df_F1.to_csv(os.path.join(F1_halfIoU_csv_path, region + ".csv"))

    return None


def multipleIoUThresh(
    root_dir, regions_coordinate_dict, platforms_list, IoU_thresh_list,
    save_qual_imgs=False, print_details=False
):
    """
    Produces csv files containing F1-scores for quantitative evaluation.
    Optionally produces visualizations of true positive, false positive, and
    false negative prediction and ground truth nuclei for each evaluation
    subregion for qualitative evaluation.
    Both of these are performed at varying intersection over union thresholds
    for F1-score calculation.

    Parameters
    root_dir: str
        the file path to the root directory. See the README.md file for details
        about the required directory structure to perform the evaluation.
    region_coordinate_dict: dict
        dictionary containing the coordinates of the top-left pixel of each
        evaluation subregion from each whole slide image. See evaluation.ipynb
        notebook for an example.
    platform_list: list
        list of strings containing the name of the segmentation platforms being
        evaluated
    IoU_thresh_list: list
        list of IoU thresholds for F1-score calculation
    save_qual_imgs: bool
        whether to save the visualizations of visualizations of true positive,
        false positive, and false negative prediction and ground truth nuclei
        for each evaluation subregion for qualitative evaluation.
    print_details: bool
        whether to print details like F1-score, TP, FP, FN, IoU threshold and
        nuclei count for each evaluation subregion of each WSI region
    """

    # creating a list of WSI fields
    regions_dir_path = os.path.join(root_dir, "Fields")
    regions_dir_files = os.listdir(regions_dir_path)
    regions_list = []
    for file in regions_dir_files:
        file_path = os.path.join(regions_dir_path, file)
        if os.path.isdir(file_path):
            regions_list.append(file)
        else:
            pass

    # Sorting the IoU thresholds in increasing order
    IoU_thresh_list.sort()

    # iterating over all WSI fields
    for region in regions_list:
        # getting file paths for platforms and ground truth binary masks
        platforms_dir_path = os.path.join(regions_dir_path, region,
                                          "Platforms")
        gr_tr_dir_path = os.path.join(regions_dir_path, region, "ground_truth")
        # getting the coordinates of the evaluation subregions for this region
        coordinate_dict = regions_coordinate_dict[region]

        # making the output directory if it does not exist
        if not os.path.exists(os.path.join(root_dir, "F1_multipleIoU_csv")):
            os.makedirs(os.path.join(root_dir, "F1_multipleIoU_csv"))
        else:
            pass
        F1_multipleIoU_csv_path = os.path.join(root_dir, "F1_multipleIoU_csv")

        # reading the ground truth binary masks
        gr_tr_sparse1 = io.imread(os.path.join(gr_tr_dir_path,
                                               "gr_tr_sparse1.tiff"))
        gr_tr_sparse2 = io.imread(os.path.join(gr_tr_dir_path,
                                               "gr_tr_sparse2.tiff"))
        gr_tr_dense1 = io.imread(os.path.join(gr_tr_dir_path,
                                              "gr_tr_dense1.tiff"))
        gr_tr_dense2 = io.imread(os.path.join(gr_tr_dir_path,
                                              "gr_tr_dense2.tiff"))

        # creating a dictionary with the ground truth binary masks
        gr_tr_mask_dict = {
            "sparse1": gr_tr_sparse1,
            "sparse2": gr_tr_sparse2,
            "dense1": gr_tr_dense1,
            "dense2": gr_tr_dense2,
        }

        # Initializing the dictionary which will contain the F1-scores
        # (averaged over 4 subfields)
        df_mean_dict = {}
        df_mean = pd.DataFrame(df_mean_dict)

        # Iterating for various IoU thresholds
        for IoU_thresh in IoU_thresh_list:
            # this dictionary will be populated with IoU thresholds and their
            # associated mean F1 scores for each platform.
            df_dict_F1 = {"subfield": []}
            # iterating over the segmentation platform masks
            for i in platforms_list:
                df_dict_F1[i] = []

            # iterating over all 4 subfields
            for subfield in gr_tr_mask_dict.keys():
                # iterating over each platform's binary mask
                for img in os.listdir(platforms_dir_path):
                    # only processed tiff files
                    if img[-5:] == ".tiff" or img[-4:] == ".tif":
                        platform_whole_img = io.imread(
                            os.path.join(platforms_dir_path, img)
                        )
                        platform = img[: img.find(".")]
                        # adds a new entry in the dictionary for the subfield
                        # if it is not already added
                        if subfield not in df_dict_F1["subfield"]:
                            df_dict_F1["subfield"].append(subfield)
                        else:
                            pass
                        # gets the ground truth mask for the current subfield
                        gr_tr_img = gr_tr_mask_dict[subfield]
                        # gets the coordinates for the current subfield
                        top_left_row = coordinate_dict[subfield][0]
                        top_left_column = coordinate_dict[subfield][1]
                        # crops the platform binary mask to isolate the region
                        # to be evaluated
                        # Evaluation subregions are 256 X 256 pixels
                        platform_img = platform_whole_img[
                            top_left_row: top_left_row + 256,
                            top_left_column: top_left_column + 256,
                        ]
                        # Not considering nuclei touching the region borders
                        platform_img = skimage.segmentation.clear_border(
                            platform_img)
                        gr_tr_img = skimage.segmentation.clear_border(
                            gr_tr_img)
                        # creating label arrays from binary data for
                        # predictions and ground truth
                        platform_img_labels = skimage.morphology.label(
                            platform_img)
                        gr_tr_img_labels = skimage.morphology.label(gr_tr_img)
                        gr_tr_img_labels = skimage.segmentation.relabel_sequential(gr_tr_img_labels)[0]
                        platform_img_labels = skimage.segmentation.relabel_sequential(platform_img_labels)[0]

                        # couting the number of predicted and ground truth
                        # nuclei
                        true_objects = len(np.unique(gr_tr_img_labels))
                        pred_objects = len(np.unique(platform_img_labels))

                        # calculating the F1-score of the current platform
                        # subfield at the current IoU threshold
                        (
                            F1,
                            nuclei_count,
                            cur_gr_tr_nuclei_count,
                            aHD,
                            matches,
                        ) = F1_score_calculator(
                            platform_img, gr_tr_img, IoU_thresh,
                            printing=print_details)
                        # to save the qualitative images for qualitative
                        # evaluation
                        # Creating outpute directories if they don't already
                        # exist
                        if save_qual_imgs:
                            if not os.path.exists(
                                os.path.join(
                                    root_dir, "qual_imgs", region,
                                    "multipleIoU"
                                )
                            ):
                                os.makedirs(
                                    os.path.join(
                                        root_dir, "qual_imgs", region,
                                        "multipleIoU"
                                    )
                                )
                            else:
                                pass
                            eval_images_multipleIoU_path = os.path.join(
                                os.path.join(
                                    root_dir, "qual_imgs", region,
                                    "multipleIoU"
                                )
                            )

                            if not os.path.exists(
                                os.path.join(
                                    eval_images_multipleIoU_path,
                                    f"{IoU_thresh}"
                                )
                            ):
                                # Create the directory
                                os.makedirs(
                                    os.path.join(
                                        eval_images_multipleIoU_path,
                                        f"{IoU_thresh}"
                                    )
                                )
                            else:
                                pass
                            if not os.path.exists(
                                os.path.join(
                                    eval_images_multipleIoU_path,
                                    f"{IoU_thresh}",
                                    "ground_truth",
                                )
                            ):
                                # Create the directory
                                os.makedirs(
                                    os.path.join(
                                        eval_images_multipleIoU_path,
                                        f"{IoU_thresh}",
                                        "ground_truth",
                                    )
                                )
                            else:
                                pass
                            if not os.path.exists(
                                os.path.join(
                                    eval_images_multipleIoU_path,
                                    f"{IoU_thresh}",
                                    "predictions",
                                )
                            ):
                                # Create the directory
                                os.makedirs(
                                    os.path.join(
                                        eval_images_multipleIoU_path,
                                        f"{IoU_thresh}",
                                        "predictions",
                                    )
                                )
                            else:
                                pass

                            # finding true positives in ground truth
                            TP_grtr = np.any(matches, axis=1)
                            TP_grtr_IDs = np.array(
                                range(1, true_objects))[TP_grtr]
                            TP_grtr_mask = np.isin(gr_tr_img_labels,
                                                   TP_grtr_IDs)
                            # finding false negatives in ground truth
                            FN_grtr = ~np.any(matches, axis=1)
                            FN_grtr_IDs = np.array(
                                range(1, true_objects))[FN_grtr]
                            FN_grtr_mask = np.isin(
                                gr_tr_img_labels, FN_grtr_IDs)
                            # finding true positives in predictions
                            TP_pred = np.any(matches, axis=0)
                            TP_pred_IDs = np.array(
                                range(1, pred_objects))[TP_pred]
                            TP_pred_mask = np.isin(
                                platform_img_labels, TP_pred_IDs)
                            # finding false positives in predictions
                            FP_pred = ~np.any(matches, axis=0)
                            FP_pred_IDs = np.array(
                                range(1, pred_objects))[FP_pred]
                            FP_pred_mask = np.isin(
                                platform_img_labels, FP_pred_IDs)

                            # initializing array for qualitative evaluation of
                            # predictions
                            pred_viz = np.zeros(
                                (platform_img.shape[0], platform_img.shape[1],
                                 3),
                                dtype=np.uint8,
                            )
                            # making true positive green and false positive red
                            # for the predictions
                            pred_viz[TP_pred_mask, :] = [0, 255, 0]
                            pred_viz[FP_pred_mask, :] = [255, 0, 0]
                            # initializing array for qualitative evaluation of
                            # ground truth
                            grtr_viz = np.zeros(
                                (gr_tr_img.shape[0], gr_tr_img.shape[1], 3),
                                dtype=np.uint8,
                            )
                            # making true positive green and false positive
                            # orange for the ground truth nuclei
                            grtr_viz[TP_grtr_mask, :] = [0, 255, 0]
                            grtr_viz[FN_grtr_mask, :] = [253, 127, 57]

                            # saving the qualitative visualizations
                            io.imsave(
                                os.path.join(
                                    eval_images_multipleIoU_path,
                                    f"{IoU_thresh}",
                                    "predictions",
                                    platform + "_" + subfield + ".tiff",
                                ),
                                pred_viz,
                                check_contrast=False,
                            )
                            io.imsave(
                                os.path.join(
                                    eval_images_multipleIoU_path,
                                    f"{IoU_thresh}",
                                    "ground_truth",
                                    platform + "_" + subfield + ".tiff",
                                ),
                                grtr_viz,
                                check_contrast=False,
                            )

                        else:
                            pass

                        # printing details if required
                        if print_details:
                            print(region + " " + platform + "_" + subfield)
                            print(f"IoU threshold: {IoU_thresh}")
                            print("***")
                        else:
                            pass

                        # Adding the F1-score to the F1-score dictionary
                        df_dict_F1[platform].append(F1)
                    else:
                        pass
            # Converting the F1-score dictionary to a dataframe
            df_F1 = pd.DataFrame(df_dict_F1)
            # Setting the index to be the subfield name
            df_F1 = df_F1.set_index("subfield")

            # Calculating the mean F1-score over the 4 subfields for each IoU
            # threshold
            df_mean[str(IoU_thresh)] = df_F1.mean()

            # converting the dataframe with the F1-scores at various IoUs to a
            # csv file
            df_mean_transpose = df_mean.transpose()
            df_mean_transpose.to_csv(
                os.path.join(F1_multipleIoU_csv_path, region + ".csv")
            )

    return None


def csv_viz_halfIoU(root_dir):
    """
    The inputs to this function are csv files containing F1-scores for
    quantitative evaluation generated using the "halfIoUThresh" function.
    A barplot is generated for visualization of the quantitative evaluation.

    Parameters
    root_dir: str
        the file path to the root directory. See the README.md file for details
        about the required directory structure to perform the evaluation.
    """

    halfIoU_csv_path = os.path.join(root_dir, "F1_halfIoU_csv")
    # creating the output directory if it doesn't already exist
    halfIoU_viz_dir = os.path.join(root_dir, "halfIoU_visaulization")
    if not os.path.exists(halfIoU_viz_dir):
        os.makedirs(halfIoU_viz_dir)
    else:
        pass

    # Appending the dataframes of each field into a list
    halfIoU_df_list = []
    for csv_file in os.listdir(halfIoU_csv_path):
        halfIoU_df_list.append(pd.read_csv(os.path.join(halfIoU_csv_path,
                                                        csv_file)))
    # Concatenating the dataframes into one big dataframe
    combined_halfIoU_df = pd.concat(halfIoU_df_list)

    ### combined_halfIoU_df = combined_halfIoU_df.rename(
    ###     columns={"CellPose": "Cellpose", "InForm": "inForm®"})

    # creating barplot with 95% confidence interval and formatting
    F1_barplot_combined = sns.barplot(combined_halfIoU_df, errorbar="ci")
    F1_barplot_combined.set_xlabel(
        xlabel="Segmentation Platform", fontweight="bold", fontsize="large"
    )
    F1_barplot_combined.set_ylabel(
        ylabel="F1-score", fontweight="bold", fontsize="large"
    )
    F1_barplot_combined.set_title(
        label="F1-score (IoU Threshold = 0.5)", fontweight="bold",
        fontsize="x-large"
    )
    plt.xticks(fontweight="regular", fontsize="large")
    plt.yticks(np.arange(0.0, 0.85, step=0.2), fontweight="regular",
               fontsize="large")

    halfIoU_viz_path = os.path.join(halfIoU_viz_dir, "halfIoU_viz.png")
    # saving plot to output directory
    plt.savefig(halfIoU_viz_path, dpi=500)

    return None


def csv_viz_multipleIoU(root_dir):
    """
    The inputs to this function are csv files containing F1-scores for
    quantitative evaluation generated using the "multipleIoUThresh" function.
    A lineplot is generated for visualization of the quantitative evaluation.

    Parameters
    root_dir: str
        the file path to the root directory. See the README.md file for details
        about the required directory structure to perform the evaluation.
    """

    multipleIoU_csv_path = os.path.join(root_dir, "F1_multipleIoU_csv")
    # creating the output directory if it doesn't already exist
    multipleIoU_viz_dir = os.path.join(root_dir, "multipleIoU_visaulization")
    if not os.path.exists(multipleIoU_viz_dir):
        os.makedirs(multipleIoU_viz_dir)
    else:
        pass

    # Appending the dataframes of each field into a list
    multipleIoU_df_list = []
    for csv_file in os.listdir(multipleIoU_csv_path):
        multipleIoU_df_list.append(
            pd.read_csv(os.path.join(multipleIoU_csv_path, csv_file))
        )
    # Concatenating the dataframes into one big dataframe
    combined_multipleIoU_df = sum(multipleIoU_df_list) / len(
        multipleIoU_df_list)
    combined_multipleIoU_df = combined_multipleIoU_df.set_index("Unnamed: 0")
    # creating lineplot and formatting
    combined_F1_lineplot = sns.lineplot(
        data=combined_multipleIoU_df, dashes=False, markers="s", legend=True
    )
    combined_F1_lineplot.set_xlabel(
        xlabel="IoU Threshold", fontweight="bold", fontsize="large"
    )
    combined_F1_lineplot.set_ylabel(
        ylabel="F1-score", fontweight="bold", fontsize="large"
    )
    combined_F1_lineplot.set_title(
        label="F1-score at Varying IoU Thresholds",
        fontweight="bold",
        fontsize="x-large",
    )
    plt.xticks(np.arange(0, 0.95, step=0.2), fontweight="regular",
               fontsize="large")
    plt.yticks(np.arange(0, 0.85, step=0.1), fontweight="regular",
               fontsize="large")

    # saving lineplot to output directory
    multipleIoU_viz_path = os.path.join(multipleIoU_viz_dir,
                                        "multipleIoU_viz.png")
    plt.savefig(multipleIoU_viz_path, dpi=500)

    return None
