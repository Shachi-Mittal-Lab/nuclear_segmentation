## CONTENTS: ##
1. About
2. Installation
3. Usage
4. Citation



## 1. ABOUT ##
- - - -
Most multiplexed immunofluorescence workflows involve segmenting individual cells from tissue samples. A nuclear marker, such as DAPI, is usually used to segment individual nuclei. The accuracy achievable by segmentation platforms is a persistent bottleneck that is preventing the advancement of the field. To make matters worse, different segmentation platforms perform with varying levels of success on different datasets due to a multitude of factors (cell morphology, illumination differences between microscopes, diffused staining etc.). There is a need for an evaluation scheme to compare nuclear segmentation performance of various platforms for a particular dataset so as to chose one to proceed with.

This repository aims at providing a means for standardized implementation, as well as qualitative and quantitative evaluation of various nuclear segmentation platforms. The code can be tweaked to customize for other platforms being evaluated.


## 2. INSTALLATION ##
- - - - 


## 3. USAGE ##
- - - - 
<b>Segmentation</b>

The first step is to perform nuclear segmentation using the DAPI channel of some sample fields of a multiplex immunofluorescence dataset. We provide detailed notebooks for running CellPose, Mesmer, and StarDist deep learning platforms. The `CellPose_segmentation.ipynb`, `Mesmer_segmentation.ipynb`, and `StarDist_segmentation.ipynb` notebooks allow users to load pre-trained deep learning models, perform nuclear segmentation, and perform necessary post-processing steps to generate nuclear binary masks for evaluation. For inForm and QuPath, morphological parameters need to be optimized, which requires experience and time. We provide general for instructions for this in `QuPath_segmentation.ipynb` and `inForm_segmentation.ipynb`.Upon exporting nuclear masks from inForm and QuPath, the notebooks also allow for generation of binary masks for evaluation. We also provide general instructions for segmentation and generation of binary masks using CellProfiler and Fiji as platforms in `CellProfiler_segmentation.ipynb` and `Fiji_segmentation.ipynb`. Users can perform the segmentation with other platforms as well and integrate their binary masks into this pipeline during `evaluation`.

<b>Evaluation</b>

The evaluation step comes after segmentation and generation of nuclear binary masks for evaluation using the platforms to be tested. The next step is the ground truth creation for evaluation. Guidance for this is in the `single_field.ipynb` notebook. The notebook also outlines the recommended file structre for storing platform-derived and ground truth binary nuclear masks for compatibility with the evaluation code. `single_field.ipynb` produces csv files which contain F1 scores at varying IoU thresholds for the field. After running `single_field.ipynb` for each field, feed in the dierctory path to the csv files in `all_fields.ipynb` which will generate evaluations for all the fields combined. Conclusions can then be drawn about which platform to proceed with for the entire dataset.

<b>Imgs</b>

We have also provided the grayscaled DAPI channels for some sample fields along with information about their tissue type in `imgs/DAPI_grayscale`. Additionally, we have coordinates for four evaluation sub-fields as well as their ground truth annotations for each of the sample field in `imgs/DAPI_grayscale/ground_truth_masks_coordinates`. These data can be used to test out the segmentation and evaluation pipelines in this repository.

Qualitatitve platform vs platform comparison on a single evaluation sub-field:
CellPose                           | Mesmer
:----: | :-----:
![Alt text](imgs/CellPose_dense2.png) | ![Alt text](imgs/Mesmer_dense2.png)


Overall quantitative comparison:
IoU=0.5 | Varying IoUs
:---: | :---:
![Alt text](imgs/melanoma_5IoU.png) | ![Alt text](imgs/melanoma_allIoU.png)


## 4. CITATION ##
- - - - 
