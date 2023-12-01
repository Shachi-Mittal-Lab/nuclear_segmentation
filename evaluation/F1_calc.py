import os
import skimage.io as io
import pandas as pd
import numpy as np
import skimage.morphology
import skimage.segmentation
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff as hausdorff
from scipy import stats
import sigfig
import seaborn as sns
from evaluation_viz import intersection_over_union
from evaluation_viz import F1_score_calculator
sns.set_style("whitegrid")

def halfIoUThresh(root_dir, regions_coordinate_dict, save_qual_imgs,
                  print_qual_imgs, platforms_list):

    regions_dir_path = os.path.join(root_dir, 'Fields')
    regions_dir_files = os.listdir(regions_dir_path)
    regions_list = []
    for file in regions_dir_files:
        file_path = os.path.join(regions_dir_path, file)
        if os.path.isdir(file_path):
            regions_list.append(file)
        else:
            pass

    for region in regions_list:

        platforms_dir_path = os.path.join(regions_dir_path, region,
                                          'Platforms')
        gr_tr_dir_path = os.path.join(regions_dir_path, region,
                                          'ground_truth')
        coordinate_dict = regions_coordinate_dict[region]

        if not os.path.exists(os.path.join(root_dir, 'F1_halfIoU_csv')):
            os.makedirs(os.path.join(root_dir, 'F1_halfIoU_csv'))
        else:
            pass
        F1_halfIoU_csv_path = os.path.join(root_dir, 'F1_halfIoU_csv')

        gr_tr_sparse1 = io.imread(os.path.join(gr_tr_dir_path,
                                               'gr_tr_sparse1.tiff'))
        gr_tr_sparse2 = io.imread(os.path.join(gr_tr_dir_path,
                                               'gr_tr_sparse2.tiff'))
        gr_tr_dense1 = io.imread(os.path.join(gr_tr_dir_path,
                                              'gr_tr_dense1.tiff'))
        gr_tr_dense2 = io.imread(os.path.join(gr_tr_dir_path,
                                              'gr_tr_dense2.tiff'))
        
        gr_tr_mask_dict = {'sparse1': gr_tr_sparse1, 'sparse2': gr_tr_sparse2,
                           'dense1': gr_tr_dense1, 'dense2': gr_tr_dense2}
        
        # this dictionary will be populated with subfield names and their associated F1 scores for each platform
        df_dict_F1 = {'subfield': []}
        for i in platforms_list:
            df_dict_F1[i] = []

        # initializing the ground truth nuclei count to be incremented
        gr_tr_nuclei_count = []
        # iterating over all 4 subfields
        for subfield in gr_tr_mask_dict.keys():
            # iterating over each platform's binary mask
            for img in os.listdir(platforms_dir_path):
                # processes only tiff files
                if img[-5:]=='.tiff' or img[-4:]=='.tif':
                    platform_whole_img = io.imread(os.path.join(platforms_dir_path, img))
                    platform = img[:img.find('.')]
                    # adds a new entry in the dictionary for the subfield if it is not already added
                    if subfield not in df_dict_F1['subfield']:
                        df_dict_F1['subfield'].append(subfield)
                    else:
                        pass
                    # gets the ground truth mask for the current subfield
                    gr_tr_img = gr_tr_mask_dict[subfield]
                    # gets the coordinates for the current subfield
                    top_left_row = coordinate_dict[subfield][0]
                    top_left_column = coordinate_dict[subfield][1]
                    # crops the platform binary mask to isolate the region to be evaluated
                    platform_img = platform_whole_img[top_left_row:top_left_row+256, top_left_column:top_left_column+256]
                    # creating a ground truth binary mask for visualization purposes in white and black
                    gr_tr_img_show = np.uint8(np.zeros((gr_tr_img.shape[0], gr_tr_img.shape[1], 3)))
                    gr_tr_img_show[gr_tr_img == 255] = [255, 255, 255]
                    # creating a ground truth binary mask for visualization purposes in green and black
                    platform_img_show = np.uint8(np.zeros((platform_img.shape[0], platform_img.shape[1], 3)))
                    platform_img_show[platform_img == 255] = [255, 99, 71]
                    # overlaying the platform over the ground truth binary mask
                    dst = cv2.addWeighted(gr_tr_img_show, 0.6, platform_img_show, 0.9, 0)
                    dst[np.all(dst == (230, 89, 64), axis=-1)] = (255, 99, 71)
                    dst[np.all(dst == (255, 242, 217), axis=-1)] = (255, 203, 203)

                    # for visualization in the notebook
                    if print_qual_imgs:
                        plt.imshow(dst)
                        plt.show()
                    else:
                        pass
                    # to save the overlayed images for qualitative evaluation
                    if save_qual_imgs:
                        if not os.path.exists(os.path.join(root_dir, 'qual_imgs', region, 'halfIoU')):
                            os.makedirs(os.path.join(root_dir, 'qual_imgs', region, 'halfIoU'))
                        else:
                            pass
                        eval_images_halfIoU_path = os.path.join(os.path.join(root_dir, 'qual_imgs', region, 'halfIoU'))
                        io.imsave(os.path.join(eval_images_halfIoU_path, platform + '_' + subfield + '.tiff'), dst, check_contrast=False)
                    else:
                        pass

                    # calculating the F1-score of the current platform subfield
                    F1, nuclei_count, cur_gr_tr_nuclei_count, aHD, matches = F1_score_calculator(platform_img, gr_tr_img, 0.5, printing=print_qual_imgs)
                    if print_qual_imgs:
                        print(platform + '_' + subfield)
                        print('----------------------------------------------')
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
        df_F1 = df_F1.set_index('subfield')

        df_F1.to_csv(os.path.join(F1_halfIoU_csv_path, region + '.csv'))

    return None


def multipleIoUThresh(root_dir, regions_coordinate_dict, save_qual_imgs,
                  platforms_list, IoU_thresh_list):
    
    regions_dir_path = os.path.join(root_dir, 'Fields')
    regions_dir_files = os.listdir(regions_dir_path)
    regions_list = []
    for file in regions_dir_files:
        file_path = os.path.join(regions_dir_path, file)
        if os.path.isdir(file_path):
            regions_list.append(file)
        else:
            pass

    assert all(0.5 <= iThresh <= 1 for iThresh in IoU_thresh_list), "Error: IoU thresholds should be between 0.5 and 1."
    IoU_thresh_list.sort()

    for region in regions_list:

        platforms_dir_path = os.path.join(regions_dir_path, region,
                                          'Platforms')
        gr_tr_dir_path = os.path.join(regions_dir_path, region,
                                          'ground_truth')
        coordinate_dict = regions_coordinate_dict[region]

        if not os.path.exists(os.path.join(root_dir, 'F1_multipleIoU_csv')):
            os.makedirs(os.path.join(root_dir, 'F1_multipleIoU_csv'))
        else:
            pass
        F1_multipleIoU_csv_path = os.path.join(root_dir, 'F1_multipleIoU_csv')

        gr_tr_sparse1 = io.imread(os.path.join(gr_tr_dir_path,
                                               'gr_tr_sparse1.tiff'))
        gr_tr_sparse2 = io.imread(os.path.join(gr_tr_dir_path,
                                               'gr_tr_sparse2.tiff'))
        gr_tr_dense1 = io.imread(os.path.join(gr_tr_dir_path,
                                              'gr_tr_dense1.tiff'))
        gr_tr_dense2 = io.imread(os.path.join(gr_tr_dir_path,
                                              'gr_tr_dense2.tiff'))
        
        gr_tr_mask_dict = {'sparse1': gr_tr_sparse1, 'sparse2': gr_tr_sparse2,
                           'dense1': gr_tr_dense1, 'dense2': gr_tr_dense2}
        
        # Initializing the dictionary which will contain the F1-scores (averaged over 4 subfields)
        df_mean_dict = {}
        df_mean = pd.DataFrame(df_mean_dict)

        # Iterating for various IoU thresholds
        for IoU_thresh in IoU_thresh_list:

            # this dictionary will be populated with IoU thresholds and their associated mean F1 scores for each platform.
            df_dict_F1 = {'subfield': []}
            for i in platforms_list:
                df_dict_F1[i] = []

            # iterating over all 4 subfields
            for subfield in gr_tr_mask_dict.keys():
                # iterating over each platform's binary mask
                for img in os.listdir(platforms_dir_path):
                    # only processed tiff files
                    if img[-5:]=='.tiff' or img[-4:]=='.tif':
                        platform_whole_img = io.imread(os.path.join(platforms_dir_path, img))
                        platform = img[:img.find('.')]
                        # adds a new entry in the dictionary for the subfield if it is not already added
                        if subfield not in df_dict_F1['subfield']:
                            df_dict_F1['subfield'].append(subfield)
                        else:
                            pass
                        # gets the ground truth mask for the current subfield
                        gr_tr_img = gr_tr_mask_dict[subfield]
                        # gets the coordinates for the current subfield
                        top_left_row = coordinate_dict[subfield][0]
                        top_left_column = coordinate_dict[subfield][1]
                        # crops the platform binary mask to isolate the region to be evaluated
                        platform_img = platform_whole_img[top_left_row:top_left_row+256, top_left_column:top_left_column+256]
                        platform_img_labels = skimage.morphology.label(platform_img)
                        gr_tr_img_labels = skimage.morphology.label(gr_tr_img)
                        gr_tr_img_labels = skimage.segmentation.relabel_sequential(gr_tr_img_labels)[0]
                        platform_img_labels = skimage.segmentation.relabel_sequential(platform_img_labels)[0]

                        true_objects = len(np.unique(gr_tr_img_labels))
                        pred_objects = len(np.unique(platform_img_labels))
                        
                        # calculating the F1-score of the current platform subfield at the current IoU threshold
                        F1, nuclei_count, cur_gr_tr_nuclei_count, aHD, matches = F1_score_calculator(platform_img, gr_tr_img,
                                                                                            IoU_thresh, printing=False)
                        # to save the overlayed images for qualitative evaluation
                        if save_qual_imgs:

                            if not os.path.exists(os.path.join(root_dir, 'qual_imgs', region, 'multipleIoU')):
                                os.makedirs(os.path.join(root_dir, 'qual_imgs', region, 'multipleIoU'))
                            else:
                                pass
                            eval_images_multipleIoU_path = os.path.join(os.path.join(root_dir, 'qual_imgs', region, 'multipleIoU'))
                            
                            if not os.path.exists(os.path.join(eval_images_multipleIoU_path, f'{IoU_thresh}')):
                                # Create the directory
                                os.makedirs(os.path.join(eval_images_multipleIoU_path, f'{IoU_thresh}'))
                            else:
                                pass
                            if not os.path.exists(os.path.join(eval_images_multipleIoU_path, f'{IoU_thresh}', 'ground_truth')):
                                # Create the directory
                                os.makedirs(os.path.join(eval_images_multipleIoU_path, f'{IoU_thresh}', 'ground_truth'))
                            else:
                                pass
                            if not os.path.exists(os.path.join(eval_images_multipleIoU_path, f'{IoU_thresh}', 'predictions')):
                                # Create the directory
                                os.makedirs(os.path.join(eval_images_multipleIoU_path, f'{IoU_thresh}', 'predictions'))
                            else:
                                pass

                            TP_grtr = np.any(matches, axis=1)
                            TP_grtr_IDs = np.array(range(1, true_objects))[TP_grtr]
                            TP_grtr_mask = np.isin(gr_tr_img_labels, TP_grtr_IDs)
                            
                            FN_grtr = ~np.any(matches, axis=1)
                            FN_grtr_IDs = np.array(range(1, true_objects))[FN_grtr]
                            FN_grtr_mask = np.isin(gr_tr_img_labels, FN_grtr_IDs)
            
                            TP_pred = np.any(matches, axis=0)
                            TP_pred_IDs = np.array(range(1, pred_objects))[TP_pred]
                            TP_pred_mask = np.isin(platform_img_labels, TP_pred_IDs)
                            
                            FP_pred = ~np.any(matches, axis=0)
                            FP_pred_IDs = np.array(range(1, pred_objects))[FP_pred]
                            FP_pred_mask = np.isin(platform_img_labels, FP_pred_IDs)

                            pred_viz = np.zeros((platform_img.shape[0], platform_img.shape[1], 3), dtype=np.uint8)
                            pred_viz[TP_pred_mask, :] = [255, 0, 0]
                            pred_viz[FP_pred_mask, :] = [0, 0, 255]
                            
                            grtr_viz = np.zeros((gr_tr_img.shape[0], gr_tr_img.shape[1], 3), dtype=np.uint8)
                            grtr_viz[TP_grtr_mask, :] = [0, 255, 0]
                            grtr_viz[FN_grtr_mask, :] = [253, 127, 57]

                            io.imsave(os.path.join(
                                eval_images_multipleIoU_path, f'{IoU_thresh}', 'predictions', platform + '_' + subfield + '.tiff'), pred_viz, check_contrast=False)
                            io.imsave(os.path.join(
                                eval_images_multipleIoU_path, f'{IoU_thresh}', 'ground_truth', platform + '_' + subfield + '.tiff'), grtr_viz, check_contrast=False)
                        
                        else:
                            pass

                        # Adding the F1-score to the F1-score dictionary
                        df_dict_F1[platform].append(F1)
                    else:
                        pass
            # Converting the F1-score dictionary to a dataframe
            df_F1 = pd.DataFrame(df_dict_F1)
            # Setting the index to be the subfield name
            df_F1 = df_F1.set_index('subfield')

            # Calculating the mean F1-score over the 4 subfields for each IoU threshold
            df_mean[str(IoU_thresh)] = df_F1.mean()

            # converting the dataframe with the F1-scores at various IoUs to a csv file
            df_mean_transpose = df_mean.transpose()
            df_mean_transpose.to_csv(os.path.join(F1_multipleIoU_csv_path, region + '.csv'))

    return None

def csv_viz_halfIoU(root_dir):
    halfIoU_csv_path = os.path.join(root_dir, 'F1_halfIoU_csv')
    halfIoU_viz_dir = os.path.join(root_dir, 'halfIoU_visaulization')
    if not os.path.exists(halfIoU_viz_dir):
        os.makedirs(halfIoU_viz_dir)
    else:
        pass
    halfIoU_viz_path = os.path.join(halfIoU_viz_dir, 'halfIoU_viz.png')

    # Appending the dataframes of each field into a list
    halfIoU_df_list = []
    for csv_file in os.listdir(halfIoU_csv_path):
        halfIoU_df_list.append(pd.read_csv(os.path.join(halfIoU_csv_path, csv_file)))
    # Concatenating the dataframes into one big dataframe
    combined_halfIoU_df = pd.concat(halfIoU_df_list)
    F1_barplot_combined = sns.barplot(combined_halfIoU_df, errorbar='ci')
    F1_barplot_combined.set(xlabel ="Platform", ylabel = "F1-score", title ='F1-score (IoU Threshold = 0.5)')
    plt.savefig(halfIoU_viz_path, dpi=500)

    return None

def csv_viz_multipleIoU(root_dir):
    multipleIoU_csv_path = os.path.join(root_dir, 'F1_multipleIoU_csv')
    multipleIoU_viz_dir = os.path.join(root_dir, 'multipleIoU_visaulization')
    if not os.path.exists(multipleIoU_viz_dir):
        os.makedirs(multipleIoU_viz_dir)
    else:
        pass
    multipleIoU_viz_path = os.path.join(multipleIoU_viz_dir, 'multipleIoU_viz.png')

    # Appending the dataframes of each field into a list
    multipleIoU_df_list = []
    for csv_file in os.listdir(multipleIoU_csv_path):
        multipleIoU_df_list.append(pd.read_csv(os.path.join(multipleIoU_csv_path, csv_file)))
    # Concatenating the dataframes into one big dataframe
    combined_multipleIoU_df = sum(multipleIoU_df_list)/len(multipleIoU_df_list)
    combined_multipleIoU_df = combined_multipleIoU_df.set_index('Unnamed: 0')
    combined_F1_lineplot = sns.lineplot(data=combined_multipleIoU_df, dashes=False, markers='s', legend=True)
    combined_F1_lineplot.set(xlabel ="IoU Threshold", ylabel = "F1-score", title ='F1-score at Varying IoU Thresholds')
    plt.savefig(multipleIoU_viz_path, dpi=500)

    return None