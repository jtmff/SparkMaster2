# SparkMaster 2 (SM2)
# Copyright (C) 2023 Jakub Tomek, jakub.tomek.mff@gmail.com
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""This is a module containing the functions providing core functionality of SparkMaster 2."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import scipy.ndimage as spnd
import os
from scipy.signal import convolve2d
from scipy.signal.windows import gaussian
from skimage.morphology import opening, disk
from skimage import measure
from skimage.measure import regionprops
import scipy.ndimage
import time
import random
from collections import deque
import cv2
import warnings
import imagesc

_TO_PRINT_TIMING = True

def get_default_parameters(pixel_width_um, lines_per_second):
    """Get default parametrization for SM2 spark analysis.

        Parameters
        ----------
        pixel_width_um : float
            width of a pixel in um (used in conversion from pixels to um)
        lines_per_second : float
            lps of the recording to be analyzed (used in conversion from pixels to ms)

        Returns
        -------
        parameters (dictionary): A dictionary where keys define the parameter names.
        """
    parameters = {}

    # Converting lps to pixel duration in ms
    parameters["pixel_width_um"] = pixel_width_um
    pixel_duration_ms = 1000/lines_per_second
    parameters["pixel_duration_ms"] = pixel_duration_ms

    parameters["extracellular_rejection"] = 0
    parameters["spark_detection_compound_threshold"] = 0.45
    # PREPROCESSING

    parameters["preprocessing_space_background_subtraction"] = True

    parameters["preprocessing_time_background_subtraction"] = True

    default_radius_um = 2
    parameters["preprocessing_1dGaussFilter_um"] = default_radius_um
    parameters["preprocessing_1dGaussFilter_pixels"] = round(default_radius_um / pixel_width_um)

    default_median_um = 1
    parameters["preprocessing_medfilt_width_um"] = default_median_um
    parameters["preprocessing_medfilt_width_pixels"] = round(default_median_um / pixel_width_um)

    default_median_ms = 16
    parameters["preprocessing_medfilt_duration_ms"] = default_median_ms
    parameters["preprocessing_medfilt_duration_pixels"] = round(default_median_um / pixel_width_um)

    default_temporal_filter_duration = 1500
    parameters["preprocessing_temporal_filter_duration"] = default_temporal_filter_duration
    parameters["preprocessing_row_normalization_filter_length_pixels"] = round(default_temporal_filter_duration/pixel_duration_ms)

    # SPARK DETECTION
    default_morpho_radius_um = 0.25
    parameters["spark_detection_morpho_radius_um"] = default_morpho_radius_um
    parameters["spark_detection_morpho_radius_pixels"] = round(default_morpho_radius_um / pixel_width_um)

    parameters["spark_detection_object_detection_threshold"] = 2.75

    parameters["spark_detection_quantile_level"] = 0.75
    #parameters["spark_detection_min_object_density"] = 0.3

    # LONG SPARK
    parameters["long_sparks_search_for_long_sparks"] = 1

    default_long_spark_core_diameter_um = 4
    parameters["long_sparks_core_diameter_um"] = default_long_spark_core_diameter_um
    parameters["long_sparks_core_diameter_pixels"] = round(default_long_spark_core_diameter_um / pixel_width_um)

    default_min_long_spark_duration_ms = 225
    parameters["long_sparks_min_long_spark_duration_ms"] = default_min_long_spark_duration_ms
    parameters["long_sparks_min_long_spark_duration_pixels"] = round(default_min_long_spark_duration_ms/pixel_duration_ms)

    default_long_spark_width_um = 1.5 * default_long_spark_core_diameter_um
    parameters["long_sparks_max_long_spark_width_um"] = default_long_spark_width_um
    parameters["long_sparks_max_long_spark_width_pixels"] = round(default_long_spark_width_um/pixel_width_um)

    parameters["long_sparks_threshold_long_sparkiness"] = 0.7

    # SPLITTING SPARKS
    parameters["splitting_sparks_splitting_threshold_step"] = 1
    parameters["splitting_sparks_splitting_min_split_depth"] = 2
    parameters["splitting_sparks_splitting_max_threshold"] = 40

    # WAVE CLASSIFICATION
    parameters["wave_classification_search_for_waves"] = 1
    parameters["wave_classification_wave_subcore_detection_threshold"] = 12.5
    parameters["wave_classification_miniwave_threshold"] = 0.1
    parameters["wave_classification_wave_threshold"] = 0.65

    # SCORING FUNCTIONS
    pixel_area = pixel_duration_ms * pixel_width_um
    default_spark_scoring_size_midpoint_umms = 42
    parameters["scoring_spark_scoring_size_midpoint_umms"] = default_spark_scoring_size_midpoint_umms
    spark_scoring_size_midpoint_pixels = round(default_spark_scoring_size_midpoint_umms/pixel_area)
    default_spark_scoring_size_slope = 3
    parameters["scoring_spark_scoring_size_params_ab"] = [spark_scoring_size_midpoint_pixels, default_spark_scoring_size_slope]
    #parameters["scoring_function_spark_scoring_size_pixels"] = lambda x: 1 - 1./(1+(x/spark_scoring_size_midpoint_pixels)**default_spark_scoring_size_slope)
    parameters["scoring_function_spark_scoring_size_pixels"] = lambda x: 1 - 1. / (1 + (x / parameters["scoring_spark_scoring_size_params_ab"][0]) ** parameters["scoring_spark_scoring_size_params_ab"][1])

    default_sub_spark_scoring_size_midpoint_umms = 14
    parameters["scoring_subspark_scoring_size_midpoint_umms"] = default_sub_spark_scoring_size_midpoint_umms
    subspark_scoring_size_midpoint_pixels = round(default_sub_spark_scoring_size_midpoint_umms / pixel_area)
    default_subspark_scoring_size_slope = 3
    parameters["scoring_subspark_scoring_size_params_ab"] = [subspark_scoring_size_midpoint_pixels, default_subspark_scoring_size_slope]
    #parameters["scoring_function_subspark_scoring_size_pixels"] = lambda x: 1 - 1. / (1 + (x / subspark_scoring_size_midpoint_pixels) ** default_spark_scoring_size_slope)
    parameters["scoring_function_subspark_scoring_size_pixels"] = lambda x: 1 - 1. / (1 + (x / parameters["scoring_subspark_scoring_size_params_ab"][0]) ** parameters["scoring_subspark_scoring_size_params_ab"][1])

    default_spark_scoring_brightness_midpoint = 1.75
    default_spark_scoring_brightness_slope = 7.5
    parameters["scoring_spark_scoring_brightness_params_ab"] = [default_spark_scoring_brightness_midpoint,
                                                                default_spark_scoring_brightness_slope]
    #parameters["scoring_function_spark_scoring_brightness"] = lambda x: 1 - 1. / (1 + (x / default_spark_scoring_brightness_midpoint) ** default_spark_scoring_brightness_slope)
    parameters["scoring_function_spark_scoring_brightness"] = lambda x: 1 - 1. / (1 + (x / parameters["scoring_spark_scoring_brightness_params_ab"][0]) ** parameters["scoring_spark_scoring_brightness_params_ab"][1])

    default_wave_scoring_size_midpoint_umms = 3850
    parameters["scoring_wave_scoring_size_midpoint_umms"] = default_wave_scoring_size_midpoint_umms
    wave_scoring_size_midpoint_pixels = round(default_wave_scoring_size_midpoint_umms/pixel_area)
    default_wave_scoring_size_slope = 3
    parameters["scoring_wave_scoring_size_params_ab"] = [wave_scoring_size_midpoint_pixels, default_wave_scoring_size_slope]
    #parameters["scoring_function_wave_scoring_size_pixels"] = lambda x: 1 - 1./(1+(x/wave_scoring_size_midpoint_pixels)**default_wave_scoring_size_slope)
    parameters["scoring_function_wave_scoring_size_pixels"] = lambda x: 1 - 1. / (1 + (x / parameters["scoring_wave_scoring_size_params_ab"][0]) ** parameters["scoring_wave_scoring_size_params_ab"][1])

    default_wave_scoring_brightness_midpoint = 10
    default_wave_scoring_brightness_slope = 5
    parameters["scoring_wave_scoring_brightness_params_ab"] = [default_wave_scoring_brightness_midpoint,
                                                               default_wave_scoring_brightness_slope]
    #parameters["scoring_function_wave_scoring_brightness"] = lambda x: 1 - 1. / (1 + (x / default_wave_scoring_brightness_midpoint) ** default_wave_scoring_brightness_slope)
    parameters["scoring_function_wave_scoring_brightness"] = lambda x: 1 - 1. / (1 + (x / parameters["scoring_wave_scoring_brightness_params_ab"][0]) ** parameters["scoring_wave_scoring_brightness_params_ab"][1])

    parameters["plotting_raw_img"] = 0
    parameters["plotting_img_col_normalized"] = 0
    parameters["plotting_img_normalized"] = 0
    parameters["plotting_img_normalized_and_smoothed"] = 0
    parameters["plotting_img_SD_transform"] = 0
    parameters["plotting_img_candidate_objects"] = 0
    parameters["plotting_img_size_raw_map"] = 0
    parameters["plotting_img_size_score_map"] = 0
    parameters["plotting_img_brightness_quantile_map"] = 0
    parameters["plotting_img_brightness_score_map"] = 0
    parameters["plotting_img_object_score_before_splitting"] = 0
    parameters["plotting_img_object_coloring_before_splitting"] = 0
    parameters["plotting_img_objects_high_threshold_wave_splitting"] = 0
    parameters["plotting_img_object_coloring_after_splitting"] = 1
    parameters["plotting_img_bounding_boxes"] = 1
    parameters["plotting_density_general"] = 0 #ap_plotting_checkbox_imgDensityPlotGeneral
    parameters["plotting_density_preceding"] = 0 #ap_plotting_checkbox_imgDensityPlotPreceding

    parameters["img_output_folder"] = ""  # where output images are saved
    parameters["img_DPI"] = 150
    return parameters


def update_pixel_parameters(parameters, pixel_width_um, lines_per_second):
    """Updates pixel-unit parameters based on um or ms parameters.
       This is best used after any um/ms parameters are changed, to maintain internal consistency of the set of parameters.

        Parameters
        ----------
        parameters : dict
            the dictionary corresponding to a structure of parameters
        pixel_width_um : float
            how many micrometers does a pixel measure across side
        lines_per_second : float
            how many lines per second does the recording to be analyzed contain

        Returns
        -------
        parameters (dictionary): Updated dictionary
        """
    # recalculates all the "_pixel" variables based on provided imaging parameters. (based on um/ms info stored in parameters
    pixel_duration_ms = 1000 / lines_per_second
    pixel_area = pixel_duration_ms * pixel_width_um

    parameters["preprocessing_1dGaussFilter_pixels"] = round(parameters["preprocessing_1dGaussFilter_um"] / pixel_width_um)
    parameters["preprocessing_medfilt_width_pixels"] = round(parameters["preprocessing_medfilt_width_um"] / pixel_width_um)
    parameters["preprocessing_medfilt_duration_pixels"] = round(parameters["preprocessing_medfilt_duration_ms"] / pixel_duration_ms)
    parameters["preprocessing_row_normalization_filter_length_pixels"] = round(parameters["preprocessing_temporal_filter_duration"] / pixel_duration_ms)
    parameters["spark_detection_morpho_radius_pixels"] = round(parameters["spark_detection_morpho_radius_um"] / pixel_width_um)

    parameters["long_sparks_core_diameter_pixels"] = round(parameters["long_sparks_core_diameter_um"] / pixel_width_um)
    parameters["long_sparks_min_long_spark_duration_pixels"] = round(parameters["long_sparks_min_long_spark_duration_ms"] / pixel_duration_ms)
    parameters["long_sparks_max_long_spark_width_pixels"] = round(parameters["long_sparks_max_long_spark_width_um"] / pixel_width_um)

    parameters["scoring_spark_scoring_size_params_ab"][0] = round(parameters["scoring_spark_scoring_size_midpoint_umms"] / pixel_area)
    parameters["scoring_subspark_scoring_size_params_ab"][0] = round(parameters["scoring_subspark_scoring_size_midpoint_umms"] / pixel_area)
    parameters["scoring_wave_scoring_size_params_ab"][0] = round(parameters["scoring_wave_scoring_size_midpoint_umms"]/pixel_area)
    return parameters


def segment_sparks(img, parameters):
    """Performs spark segmentation on a given image, using the provided dict of parameters.

    A default set of parameters may be defined using SM2.get_default_parameters()

        Parameters
        ----------
        img : numpy array
            a numerical representation of the image to be analyzed (ideally uint8 or uint 16)
        parameters : dict
            the dictionary corresponding to a structure of parameters

        Returns
        -------
        numbered_mask : numpy array
            a 'numbered mask' describing the segmentation. This is a numpy array with the same size as the input image containing i's in the positions of object with the label i
        img_smoothed : numpy array
            the input image after all the image filtering/processing steps
        img_smoothed_not_rcnormalized : numpy array
            the input image after denoising, but without row/column normalization. This is an useful input to SM2.analyze_sparks when one wants e.g. the amplitudes expressed as F/F0.
        img_SD_multiple : numpy array
            the SD-transformed version of the previous output image. This gives, for each pixel, how many standard deviation (across the whole image's pixels) the pixel is from the image's mean intensity.
        numbers_waves : list
            the list of object labels classified as waves
        numbers_miniwaves : list
            the list of object labels classified as miniwaves
        numbers_long_sparks : list
            the list of object labels classified as long sparks
        """

    # Checking that the image is single-channel
    if len(img.shape) > 2:
        raise Exception("The input image has to be single-channel (grayscale). The provided image is an RGB image, which needs to be converted to a single-channel first.")

    # Defining functions based on the structure of parameters.
    function_spark_scoring_size_pixels = lambda x: 1 - 1. / (
                1 + (x / parameters["scoring_spark_scoring_size_params_ab"][0]) ** parameters["scoring_spark_scoring_size_params_ab"][1])

    function_subspark_scoring_size_pixels = lambda x: 1 - 1. / (
                1 + (x / parameters["scoring_subspark_scoring_size_params_ab"][0]) ** parameters["scoring_subspark_scoring_size_params_ab"][1])

    function_spark_scoring_brightness = lambda x: 1 - 1. / (
                1 + (x / parameters["scoring_spark_scoring_brightness_params_ab"][0]) ** parameters["scoring_spark_scoring_brightness_params_ab"][1])

    function_wave_scoring_size_pixels = lambda x: 1 - 1. / (
                1 + (x / parameters["scoring_wave_scoring_size_params_ab"][0]) ** parameters["scoring_wave_scoring_size_params_ab"][1])

    function_wave_scoring_brightness = lambda x: 1 - 1. / (
                1 + (x / parameters["scoring_wave_scoring_brightness_params_ab"][0]) ** parameters["scoring_wave_scoring_brightness_params_ab"][1])

    t0 = time.time()
    ## Calculating extracellular rejection
    threshold_for_rejection = 0.25 # Columns with at least this proportion of sub-threshold pixels are suitable for discarding
    fraction_extracellular = sum(img <= parameters["extracellular_rejection"]) / img.shape[0]
    to_reject = fraction_extracellular > threshold_for_rejection
    if sum(to_reject) + 1 >= img.shape[1]:
        raise Exception('The extracellular space threshold is too high for this recording, leading to the whole image getting discarded. Please use a lower value (a minimum of 0).')
    cutting_happens = 0

    pre_cell_starting = -1
    if to_reject[0] > 0:  # we are rejecting something at the start of the cell - search for the last point of initial segment of columns to reject
        for i_column in range(0, img.shape[1]):
            if to_reject[i_column] == 0: # if we suddenly encounter a zero, we know the previous index was the last 1
                pre_cell_starting = i_column - 1
                cutting_happens = 1
                break

    post_cell_ending = img.shape[1]+1
    if to_reject[len(fraction_extracellular)-1] > 0:  # we are rejecting something at the end of the cell - search for the first point of trailing segment of columns to reject
        for i_column in range(img.shape[1]-1, -1, -1):
            if to_reject[i_column] == 0:  # if we suddenly encounter a zero, we know the previous index was the last 1
                post_cell_ending = i_column + 1
                cutting_happens = 1
                break

    if cutting_happens:
        img_pre_cut = img
        img = img[:, (pre_cell_starting+1):(post_cell_ending-1)] # in python, the 2nd element in range is non-inclusive, hence 1 larger than naturally expected
    t1 = time.time()

    if _TO_PRINT_TIMING:
        print('time cutting = ' + str(t1-t0))
    ## Plotting the image
    if parameters["plotting_raw_img"]:
        if cutting_happens == 0:
            fig = plt.figure(dpi=parameters["img_DPI"])
            plt.imshow(img, cmap="gray")
            plt.title("Original image")
            plt.grid(False)
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle("Original image")
            ax1.imshow(img_pre_cut, cmap="gray")
            ax1.set_title("Before removing extracellular space")
            ax2.imshow(img, cmap="gray")
            ax2.set_title("After removing extracellular space")
            ax1.grid(False)
            ax2.grid(False)

        if parameters["img_output_folder"]:
            plt.savefig(os.path.join(parameters["img_output_folder"], "rawImg.png"), bbox_inches='tight')
    t2 = time.time()
    if _TO_PRINT_TIMING:
        print('time plotting raw image = ' + str(t2 - t1))

    ## Column normalization
    n_rows = img.shape[0]
    n_cols = img.shape[1]

    if parameters["preprocessing_space_background_subtraction"]:
        col_medians = np.zeros(n_cols)

        for i_col in range(n_cols):
            col_data = img[:, i_col]
            #col_data = col_data[col_data > parameters["extracellular_rejection"]]
            col_mean = np.nanmean(col_data)
            col_std = np.std(col_data)
            col_data_2 = col_data[col_data <= col_mean + 1.75*col_std]
            col_mean_2 = np.nanmean(col_data_2)
            col_std_2 = np.nanstd(col_data_2)
            col_data_filtered = col_data_2[col_data_2 <= col_mean_2 + 1.75*col_std_2]
            col_medians[i_col] = np.nanmedian(col_data_filtered)

        # Treating a special case when there are some zero column medians
        if col_medians[i_col] == 0:
            col_medians[i_col] = 1
        for i_col in range(n_cols):
            if col_medians[i_col] == 0:
                col_medians[i_col] = col_medians[i_col-1]
        # Normalizing the image
        img_col_normalized = img / col_medians[:, None].T

        if parameters["plotting_img_col_normalized"]:
            fig = plt.figure(dpi=parameters["img_DPI"])
            plt.imshow(img_col_normalized, cmap="gray")
            plt.title("Column-normalized image")
            plt.grid(False)
            if parameters["img_output_folder"]:
                plt.savefig(os.path.join(parameters["img_output_folder"], "imgColNormalized.png"), bbox_inches='tight')
    else:
        img_col_normalized = img

    t3 = time.time()
    if _TO_PRINT_TIMING:
        print('column normalization = ' + str(t3 - t2))

    ## Row normalization
    if parameters["preprocessing_time_background_subtraction"]:
        row_medians = np.zeros(n_rows)

        for i_row in range(n_rows):
            row_data = img_col_normalized[i_row, :]
            row_mean = np.nanmean(row_data)
            row_std = np.nanstd(row_data)
            row_data_2 = row_data[row_data <= row_mean + 1.75*row_std]
            row_mean_2 = np.nanmean(row_data_2)
            row_std_2 = np.nanstd(row_data_2)
            row_data_filtered = row_data_2[row_data_2 <= row_mean_2 + 1.75*row_std_2]
            row_medians[i_row] = np.nanmedian(row_data_filtered)

        norm_factor = spnd.median_filter(row_medians, (parameters["preprocessing_row_normalization_filter_length_pixels"],), mode='nearest')
        norm_factor = vector_mean(norm_factor, parameters["preprocessing_row_normalization_filter_length_pixels"])
        img_normalized = img_col_normalized / norm_factor[None, :].T

        if parameters["plotting_img_normalized"]:
            fig = plt.figure(dpi=parameters["img_DPI"])
            plt.imshow(img_normalized, cmap="gray")
            plt.title("Row and column normalized image")
            plt.grid(False)
            if parameters["img_output_folder"]:
                plt.savefig(os.path.join(parameters["img_output_folder"], "imgNormalized.png"), bbox_inches='tight')
    else:
        img_normalized = img_col_normalized

    t4 = time.time()
    if _TO_PRINT_TIMING:
        print('row normalization = ' + str(t4 - t3))

    ## Spatial filtering
    img_den = medfilt2_padded(img_normalized, (parameters["preprocessing_medfilt_duration_pixels"], parameters["preprocessing_medfilt_width_pixels"]), 'symmetric')
    gauss_filter_1d = gaussian(parameters["preprocessing_1dGaussFilter_pixels"], (parameters["preprocessing_1dGaussFilter_pixels"]-1)/5) # 2nd parameter is made to match Matlab where this was first implemented
    gauss_filter_1d = gauss_filter_1d / sum(gauss_filter_1d)
    gauss_filter_1d.shape = (1, gauss_filter_1d.shape[0])
    img_den_gauss = convolve2d(img_den, gauss_filter_1d, 'same', 'symm')

    # Making also a version of the image without row/column normalization - used in feature extraction
    img_smoothed_not_rcnormalized = medfilt2_padded(img, (parameters["preprocessing_medfilt_duration_pixels"], parameters["preprocessing_medfilt_width_pixels"]), 'symmetric')
    img_smoothed_not_rcnormalized = convolve2d(img_smoothed_not_rcnormalized, gauss_filter_1d, 'same', 'symm')


    if parameters["plotting_img_normalized_and_smoothed"]:
        fig = plt.figure(dpi=parameters["img_DPI"])
        plt.imshow(img_den_gauss, cmap="gray")
        plt.title("Smoothed and normalized image")
        plt.grid(False)
        if parameters["img_output_folder"]:
            plt.savefig(os.path.join(parameters["img_output_folder"], "imgNormalizedSmoothed.png"), bbox_inches='tight')

    t5 = time.time()
    if _TO_PRINT_TIMING:
        print('spatial filtering = ' + str(t5 - t4))

    ## SD-transform of the image
    std_cutoff = 1

    # Twice we remove objects that are a certain degree above the mean - first removing spark cores, second it removes their boundaries

    img_no_foreground = img_den_gauss[img_den_gauss <= np.mean(img_den_gauss) + std_cutoff * np.std(img_den_gauss)]
    img_no_foreground_2 = img_no_foreground[img_no_foreground <= np.mean(img_no_foreground) + std_cutoff * np.std(img_no_foreground)]




    bkg_mean = np.mean(img_no_foreground_2)
    bkg_std = np.std(img_no_foreground_2)

    img_SD_multiple = (img_den_gauss - bkg_mean) / bkg_std

    t6 = time.time()
    if _TO_PRINT_TIMING:
        print('sd transform = ' + str(t6 - t5))

    if parameters["plotting_img_SD_transform"]:
        fig = plt.figure(dpi=parameters["img_DPI"])
        plt.imshow(img_SD_multiple, cmap="viridis")
        plt.title("Stdev-transformation of the image")
        plt.grid(False)
        plt.colorbar()
        if parameters["img_output_folder"]:
            plt.savefig(os.path.join(parameters["img_output_folder"], "imgSDtransform.png"), bbox_inches='tight')

    ## Getting candidate objects/sparks

    # initial detection - still a lot of spurious objects
    mask_objects_low_thresh = img_den_gauss > bkg_mean + bkg_std * parameters["spark_detection_object_detection_threshold"]

    # morpho opening
    mask_objects_low_thresh_opened = opening(mask_objects_low_thresh, disk(parameters["spark_detection_morpho_radius_pixels"]))

    # and adding extra filter based on pixel density
    #filter_size = round(parameters["preprocessing_spark_radius_pixels"]/2)
    #K = 1/(filter_size * filter_size) * np.ones((filter_size, filter_size))
    #averaged_mask = convolve2d(mask_objects_low_thresh_opened, K, 'same', 'symm')

    mask_objects_low_thresh_filtered = mask_objects_low_thresh_opened#averaged_mask > parameters["spark_detection_min_object_density"]

    if parameters["plotting_img_candidate_objects"]:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle("Generation of candidate objects")
        ax1.imshow(mask_objects_low_thresh, cmap="gray")
        ax1.set_title("Raw candidate objects")

        ax2.imshow(mask_objects_low_thresh_opened, cmap="gray")
        ax2.set_title("After morphological opening")

        ax1.grid(False)
        ax2.grid(False)
        if parameters["img_output_folder"]:
            plt.savefig(os.path.join(parameters["img_output_folder"], "imgCandidateObjects.png"), bbox_inches='tight')

    # We use the filtered mask from now on
    mask_objects_low_thresh = mask_objects_low_thresh_filtered

    ## Getting the cell numbering
    #numbered_mask_low_thresh = measure.label(mask_objects_low_thresh > 0)
    numbered_mask_low_thresh = scipy.ndimage.label(mask_objects_low_thresh > 0)[0]
    n_candidates = np.max(numbered_mask_low_thresh)

    #indices_candidates = find_multi(numbered_mask_low_thresh)
    props_nmlt = regionprops(numbered_mask_low_thresh)
    t7 = time.time()
    if _TO_PRINT_TIMING:
        print('getting numbering = ' + str(t7 - t6))

    ## Detecting long sparks

    next_free_number = n_candidates + 1
    numbers_long_sparks = [] # Numbers of long sparks assigned in the numbered mask. We want to store this explicitly given the zero-based numbering used in Python.

    if parameters["long_sparks_search_for_long_sparks"] == 1:
        for i_object in range(0, n_candidates):
            coords = props_nmlt[i_object].coords
            label = props_nmlt[i_object].label
            rows = coords[:, 0]
            cols = coords[:, 1]
            min_r = np.min(rows)
            max_r = np.max(rows)

            if (max_r - min_r < parameters["long_sparks_min_long_spark_duration_pixels"]):
                continue  # If the detected object is temporally short, we skip it right away

            min_c = np.max([0, np.min(cols)])
            max_c = np.min([numbered_mask_low_thresh.shape[1] - 1, np.max(cols)])

            sub_image = img_SD_multiple[min_r:(max_r + 1), min_c:(max_c + 1)]
            mask_spark = np.zeros(numbered_mask_low_thresh.shape)
            mask_spark[rows, cols] = 1
            mask_spark = mask_spark[min_r:(max_r + 1), min_c:(max_c + 1)] # submask with only the image in place
            sub_numbered_mask_low_thresh = numbered_mask_low_thresh[min_r:(max_r + 1), min_c:(max_c + 1)]

            q25_sub_image = np.quantile(sub_image, 0.25, axis=0) # we take the 25-quantile of each column
            indicator_over_threshold = 1 * (q25_sub_image > 2)  # 1* for conversion to int for regionprops
            indicator_over_threshold.shape = (1, len(indicator_over_threshold))
            objects_over_threshold = regionprops(indicator_over_threshold)


            if (len(objects_over_threshold) == 1):  # we continue only of there is a single prominent spike
                width_column = np.sum(indicator_over_threshold) # length of the object
                if width_column <= parameters["long_sparks_max_long_spark_width_pixels"]: # the object needs to be not too wide - very wide objects are unlikely to be long sparks
                    indices_max = np.argmax(sub_image, axis=1) # Find in which
                    center_point_column = round((objects_over_threshold[0].coords[0, 1] + objects_over_threshold[0].coords[width_column-1, 1])/2)
                    center_left_boundary = np.max([0, center_point_column - round(parameters["long_sparks_core_diameter_pixels"]/2)])
                    center_right_boundary = np.min([sub_image.shape[1]-1, center_point_column + round(parameters["long_sparks_core_diameter_pixels"] / 2)])

                    long_sparkiness = sum(np.logical_and(indices_max >= center_left_boundary, indices_max <= center_right_boundary)) / sub_image.shape[0]
                    if long_sparkiness < parameters["long_sparks_threshold_long_sparkiness"]:
                        continue

                    col_from_long_spark = np.max([0, center_point_column - round(parameters["long_sparks_core_diameter_pixels"]/2)])
                    col_to_long_spark = np.min([sub_image.shape[1]-1, center_point_column + round(parameters["long_sparks_core_diameter_pixels"]/2)])

                    # erase the long spark from the spark mask (this may leave some remaining chunks that we will process later)
                    mask_spark[:, col_from_long_spark:(col_to_long_spark + 1)] = 0

                    # handle the remaining objects
                    #objects_remaining = regionprops(measure.label(mask_spark > 0))
                    objects_remaining = regionprops(scipy.ndimage.label(mask_spark > 0)[0])
                    if len(objects_remaining) > 0:
                        for i_remaining in range(len(objects_remaining)):
                            remaining_coords = objects_remaining[i_remaining].coords
                            remaining_rows = remaining_coords[:, 0]
                            remaining_cols = remaining_coords[:, 1]
                            sub_numbered_mask_low_thresh[remaining_rows, remaining_cols] = next_free_number
                            next_free_number = next_free_number + 1

                    numbered_mask_low_thresh[min_r:(max_r + 1), min_c:(max_c + 1)] = sub_numbered_mask_low_thresh
                    numbers_long_sparks.append(label)

    numbered_mask = numbered_mask_low_thresh

    t8 = time.time()
    if _TO_PRINT_TIMING:
        print('long spark detection = ' + str(t8 - t7))


    ## Scoring detected objects
    size_raw_map = np.zeros(numbered_mask.shape)
    size_score_map = np.zeros(numbered_mask.shape)
    quantile_brightness_raw_map = np.zeros(numbered_mask.shape)
    quantile_brightness_score_map = np.zeros(numbered_mask.shape)

    brightness_score_wave = np.zeros(next_free_number)  # for each object, we calculate the wave brightness score
    size_score_wave = np.zeros(next_free_number)
    object_compound_score_wave = np.zeros(next_free_number)
    object_to_number = np.zeros(next_free_number)
    number_to_object = np.zeros(next_free_number)

    object_to_pixels = regionprops(numbered_mask)

    for i_object in range(next_free_number - 1):
        coordinates = object_to_pixels[i_object].coords
        object_label = object_to_pixels[i_object].label
        rows = coordinates[:, 0]
        cols = coordinates[:, 1]

        # scoring size
        n_pixels = len(rows)
        size_raw_map[rows, cols] = n_pixels
        size_score_map[rows, cols] = function_spark_scoring_size_pixels(n_pixels)

        # scoring brightness
        quantile_brightness = np.quantile(img_SD_multiple[rows, cols] - parameters["spark_detection_object_detection_threshold"], parameters["spark_detection_quantile_level"])  # first subtraction can probably happen later
        quantile_brightness_raw_map[rows, cols] = quantile_brightness
        quantile_brightness_score_map[rows, cols] = function_spark_scoring_brightness(quantile_brightness)

        # computing wave scores

        brightness_score_wave[i_object] = function_wave_scoring_brightness(quantile_brightness)
        size_score_wave[i_object] = function_wave_scoring_size_pixels(n_pixels)
        object_compound_score_wave[i_object] = brightness_score_wave[i_object] * size_score_wave[i_object]
        object_to_number[i_object] = object_label
        number_to_object[object_label] = i_object
        #brightness_score_wave[object_label] = function_wave_scoring_brightness(quantile_brightness)
        #[object_label] = function_wave_scoring_size_pixels(n_pixels)
        #object_compound_score_wave[object_label] = brightness_score_wave[object_label] * size_score_wave[object_label]

    compound_score_map = np.multiply(size_score_map, quantile_brightness_score_map)

    t9 = time.time()
    if _TO_PRINT_TIMING:
        print('scoring objects = ' + str(t9 - t8))

    #parameters["plotting_img_size_raw_map"] = 0
    #parameters["plotting_img_size_score_map"] = 0
    #parameters["plotting_img_brightness_quantile_map"] = 0
    #parameters["plotting_img_brightness_score_map"] = 0
    #parameters["plotting_img_object_score_before_splitting"] = 0
    #parameters["plotting_img_object_coloring_before_splitting"] = 0
    cmap_zero_black = plt.cm.get_cmap("viridis").copy()
    cmap_zero_black.set_bad(color="black")

    # versions of images with nans instead of 0s, useful for plotting.

    srm_nanned = size_raw_map.copy()
    srm_nanned[srm_nanned == 0] = np.nan
    ssm_nanned = size_score_map.copy()
    ssm_nanned[ssm_nanned == 0] = np.nan
    qbrm_nanned = quantile_brightness_raw_map.copy()
    qbrm_nanned[qbrm_nanned == 0] = np.nan
    qbsm_nanned = quantile_brightness_score_map.copy()
    qbsm_nanned[qbsm_nanned == 0] = np.nan
    csm_nanned = compound_score_map.copy()
    csm_nanned[csm_nanned == 0] = np.nan

    if parameters["plotting_img_size_raw_map"]:
        fig = plt.figure(dpi=parameters["img_DPI"])
        plt.imshow(srm_nanned, cmap=cmap_zero_black)
        plt.title("Object size (pixels)")
        plt.grid(False)
        plt.colorbar()
        if parameters["img_output_folder"]:
            plt.savefig(os.path.join(parameters["img_output_folder"], "imgSizeRawMap.png"), bbox_inches='tight')

    if parameters["plotting_img_size_score_map"]:
        fig = plt.figure(dpi=parameters["img_DPI"])
        plt.imshow(ssm_nanned, cmap=cmap_zero_black)
        plt.title("Object size score")
        plt.grid(False)
        plt.colorbar()
        if parameters["img_output_folder"]:
            plt.savefig(os.path.join(parameters["img_output_folder"], "imgSizeScoreMap.png"), bbox_inches='tight')

    if parameters["plotting_img_brightness_quantile_map"]:
        fig = plt.figure(dpi=parameters["img_DPI"])
        plt.imshow(qbrm_nanned, cmap=cmap_zero_black)
        plt.title("Object brightness quantile")
        plt.grid(False)
        plt.colorbar()
        if parameters["img_output_folder"]:
            plt.savefig(os.path.join(parameters["img_output_folder"], "imgBrightnessQuantileMap.png"), bbox_inches='tight')

    if parameters["plotting_img_brightness_score_map"]:
        fig = plt.figure(dpi=parameters["img_DPI"])
        plt.imshow(qbsm_nanned, cmap=cmap_zero_black)
        plt.title("Object brightness score")
        plt.grid(False)
        plt.colorbar()
        if parameters["img_output_folder"]:
            plt.savefig(os.path.join(parameters["img_output_folder"], "imgBrightnessScoreMap.png"), bbox_inches='tight')

    if parameters["plotting_img_object_score_before_splitting"]:
        fig = plt.figure(dpi=parameters["img_DPI"])
        plt.imshow(csm_nanned, cmap=cmap_zero_black)
        plt.title("Object compound score")
        plt.grid(False)
        plt.colorbar()
        if parameters["img_output_folder"]:
            plt.savefig(os.path.join(parameters["img_output_folder"], "imgObjectScoreBeforeSplitting.png"), bbox_inches='tight')

    if parameters["plotting_img_object_coloring_before_splitting"]:
        fig = plt.figure(dpi=parameters["img_DPI"])
        rgb_coloring = numbered_mask_to_RGB(numbered_mask)
        plt.imshow(rgb_coloring)
        plt.title("Object coloring before splitting & low-score removal")
        plt.grid(False)

        if parameters["img_output_folder"]:
            plt.savefig(os.path.join(parameters["img_output_folder"], "imgObjectColoringBeforeSplitting.png"), bbox_inches='tight')

    t10 = time.time()
    if _TO_PRINT_TIMING:
        print('plotting = ' + str(t10 - t9))
    # removing poorly scoring objects
    numbered_mask[compound_score_map < parameters["spark_detection_compound_threshold"]] = 0

    ## Splitting waves and miniwaves, regrading them in the process
    waves = [] # list of waves (their numbers in the object list)
    miniwaves = []

    if parameters["wave_classification_search_for_waves"]:
        # this finds indices in the list of objects that correspond to waves
        i_waves = np.argwhere(object_compound_score_wave > parameters["wave_classification_wave_threshold"] ) # indices of waves are obtained (this is zero-based, it is NOT the number in the numbered mask - this is one smaller)
        i_miniwaves = np.argwhere(object_compound_score_wave > parameters["wave_classification_miniwave_threshold"])

        # now we extract the numbers of the waves
        numbers_waves = object_to_number[i_waves]
        numbers_miniwaves = object_to_number[i_miniwaves]

        # and remove long spark numbering
        numbers_waves = set(numbers_waves.flatten())-set(numbers_long_sparks)
        numbers_miniwaves = set(numbers_miniwaves.flatten()) - set(numbers_long_sparks) - numbers_waves

        mask_objects_high_thresh = img_den_gauss > (bkg_mean + parameters["wave_classification_wave_subcore_detection_threshold"] * bkg_std)
        mask_objects_high_thresh_opened = opening(mask_objects_high_thresh, disk(parameters["spark_detection_morpho_radius_pixels"]))

        #props_high_thresh = regionprops(measure.label(mask_objects_high_thresh_opened))
        props_high_thresh = regionprops(measure.label(scipy.ndimage.label(mask_objects_high_thresh_opened > 0)[0]))

        # constructing numbered mask using yet unused numbers for future convenience. 
        numbered_mask_high_thresh = np.zeros(mask_objects_high_thresh_opened.shape)
        for i_ht in range(len(props_high_thresh)):
            coords = props_high_thresh[i_ht].coords
            rows = coords[:, 0]
            cols = coords[:, 1]
            numbered_mask_high_thresh[rows, cols] = next_free_number + i_ht
            next_free_number = next_free_number + 1

        if parameters["plotting_img_objects_high_threshold_wave_splitting"]:
            fig = plt.figure(dpi=parameters["img_DPI"])
            plt.imshow(mask_objects_high_thresh_opened, cmap="gray")
            plt.title("Mask of high-threshold objects used in wave splitting")
            plt.grid(False)
            plt.colorbar()
            if parameters["img_output_folder"]:
                plt.savefig(os.path.join(parameters["img_output_folder"], "imgObjectsHighThresholdWaveSplitting.png"), bbox_inches='tight')

        # 1) We go over all objects that are waves and miniwaves.
        # 2) If it has 2 or more objects in mask_objects_high_thresh_opened, we:
        # 2.1) Competitively grow the objects
        # 2.2) Rate them, saving them as waves or miniwaves if scoring enough (no need for recursive splitting here).
        queue = list(numbers_waves) + list(numbers_miniwaves)

        for n_wave in range(len(queue)):
            n_object = int(queue[n_wave]) # number of wave
            i_object = int(number_to_object[n_object]) # index in list of found objects

            which_pixels = object_to_pixels[i_object].coords
            rows = which_pixels[:, 0]
            cols = which_pixels[:, 1]
            which_cells_high_mask = set(numbered_mask_high_thresh[rows, cols].flatten())
            which_cells_high_mask = which_cells_high_mask -set([0])

            if len(which_cells_high_mask) > 1:
                if n_object in numbers_waves:
                    numbers_waves = numbers_waves - set([n_object])
                elif n_object in numbers_miniwaves:
                    numbers_miniwaves = numbers_miniwaves - set([n_object])
                else:
                    print('the code should not get here...')

                # We do splitting here

                # First, we get the mask with high objects, but only where our currently processed [mini]wave is.
                mask_only_low_thresh_object = np.zeros(numbered_mask_high_thresh.shape)
                mask_only_low_thresh_object[rows, cols] = numbered_mask_high_thresh[rows, cols]
                mask_only_low_thresh_object_binary = mask_only_low_thresh_object > 0

                # For each object in the high-thresh mask, we make one "queue". This will be expanded with pixels that
                # 1) are 1s in the low-thresh mask, 2) are not yet filled with a particular number, 3) are down the brightness gradient
                indicator_erosion = scipy.ndimage.binary_erosion(mask_only_low_thresh_object.tolist())
                boundary_mask = mask_only_low_thresh_object.copy()
                boundary_mask[indicator_erosion] = 0
                rp_boundaries = regionprops(boundary_mask.astype(int))
                number_of_component = np.zeros((len(rp_boundaries), 1)) # number of i-th wave
                boundaries = []

                for i_boundary in range(len(rp_boundaries)):
                    coords = rp_boundaries[i_boundary].coords
                    boundaries.append(coords)
                    number_of_component[i_boundary] = int(numbered_mask_high_thresh[coords[0, 0], coords[0, 1]])

                while sum([len(boundary) for boundary in boundaries]) > 0: # while at least one of the boundaries is not empty
                    for i_boundary in range(len(rp_boundaries)): # for each boundary
                        boundary = boundaries[i_boundary]
                        new_boundary = []
                        for i_point in range(boundary.shape[0]): # for each point in the boundary
                            point_row = boundary[i_point, 0]
                            point_col = boundary[i_point, 1]
                            neighbours = []
                            if point_row > 0:
                                neighbours.append((point_row-1, point_col)) #

                            if point_row < n_rows-1:
                                neighbours.append((point_row+1, point_col))

                            if point_col > 0:
                                neighbours.append((point_row, point_col - 1))

                            if point_col < n_cols-1:
                                neighbours.append((point_row, point_col + 1))

                            for i_neighbour in range(len(neighbours)):
                                rn = neighbours[i_neighbour][0]
                                rc = neighbours[i_neighbour][1]

                                # We add the point to the new boundary if a) the new pixel is not yet assigned, b) it is within the low-thresh object, c) it is not brighter than the current point.

                                if (numbered_mask_high_thresh[rn, rc] == 0) & (numbered_mask[rn, rc] > 0) & (img_SD_multiple[point_row, point_col] >= img_SD_multiple[rn, rc]):
                                    numbered_mask_high_thresh[rn, rc] = number_of_component[i_boundary]
                                    new_boundary.append((rn, rc))

                        boundaries[i_boundary] = np.array(new_boundary)

                numbered_mask[rows, cols] = numbered_mask_high_thresh[rows, cols] # using the new numbering
                #brightness_which_pixels = img_SD_multiple[rows, cols]

                # Now we go over all the formed components, getting their locations and scoring them
                for i_component in range(len(number_of_component)):
                    where_component = np.argwhere(numbered_mask == number_of_component[i_component])
                    brightness_quantile = np.quantile(img_SD_multiple[where_component[:, 0], where_component[:, 1]] - parameters["spark_detection_object_detection_threshold"], parameters["spark_detection_quantile_level"]) 
                    brightness_score_wave = function_wave_scoring_brightness(brightness_quantile)
                    size_score_wave = function_wave_scoring_size_pixels(where_component.shape[0])
                    object_compound_score_wave = brightness_score_wave * size_score_wave
                    if (object_compound_score_wave > parameters["wave_classification_wave_threshold"]):
                        numbers_waves.add(int(number_of_component[i_component]))
                    elif (object_compound_score_wave > parameters["wave_classification_miniwave_threshold"]):
                        numbers_miniwaves.add(int(number_of_component[i_component]))
    else:
        numbers_waves = []
        numbers_miniwaves = []

    t11 = time.time()
    if _TO_PRINT_TIMING:
        print('wave splitting = ' + str(t11 - t10))

    ## Standard splitting

    objects_no_split = list(numbers_waves) + list(numbers_miniwaves) + numbers_long_sparks
    numbered_mask_after_split = split_sparks_intensity(numbered_mask, img_SD_multiple, parameters["splitting_sparks_splitting_threshold_step"], parameters["splitting_sparks_splitting_min_split_depth"], parameters["splitting_sparks_splitting_max_threshold"],
                                                       function_spark_scoring_brightness, function_subspark_scoring_size_pixels, parameters["spark_detection_compound_threshold"],
                                                       parameters["spark_detection_quantile_level"], objects_no_split)
    t12 = time.time()
    if _TO_PRINT_TIMING:
        print('spark splitting = ' + str(t12 - t11))
    ## Regrading and deleting weak objects, also calculating bounding boxes

    if parameters["plotting_img_object_coloring_after_splitting"]:
        fig = plt.figure(dpi=parameters["img_DPI"])
        rgb_coloring = numbered_mask_to_RGB(numbered_mask_after_split)
        plt.imshow(rgb_coloring)
        plt.title("Object coloring after splitting")
        plt.grid(False)

        if parameters["img_output_folder"]:
            plt.savefig(os.path.join(parameters["img_output_folder"], "imgObjectColoringAfterSplitting.png"), bbox_inches='tight')

    object_to_pixels = regionprops(numbered_mask_after_split)

    bounding_boxes = []  # list of quadruples determining bounding boxes
    box_ids = []  # id of each object

    for i_object in range(len(object_to_pixels)):
        coordinates = object_to_pixels[i_object].coords
        label = object_to_pixels[i_object].label
        rows = coordinates[:, 0]
        cols = coordinates[:, 1]

        # scoring size
        n_pixels = len(rows)
        size_score = function_spark_scoring_size_pixels(n_pixels)

        # scoring brightness
        quantile_brightness = np.quantile(img_SD_multiple[rows, cols] - parameters["spark_detection_object_detection_threshold"],
                                          parameters["spark_detection_quantile_level"])  # first subtraction can probably happen later

        quantile_brightness_score = function_spark_scoring_brightness(quantile_brightness)

        compound_score = quantile_brightness_score * size_score
        if compound_score <= parameters["spark_detection_compound_threshold"]:  # If too weak, we remove it
            numbered_mask_after_split[rows, cols] = 0
        else:  #Otherwise we record its bounding box
            min_row = min(rows)
            max_row = max(rows)
            min_col = min(cols)
            max_col = max(cols)
            bounding_boxes.append((min_col, min_row, (max_col-min_col), (max_row-min_row)))
            box_ids.append(label)

    if parameters["plotting_img_bounding_boxes"]:
        fig = plt.figure(dpi=parameters["img_DPI"])
        idg = img_den_gauss.copy()
        idg = idg-np.min(idg)
        idg = idg/np.max(idg)
        idg = idg * 255
        idg = idg.astype(np.uint8)
        img_bb = cv2.merge([idg,idg,idg])

        for i_object in range(len(bounding_boxes)):
            x, y, w, h = bounding_boxes[i_object]
            label = box_ids[i_object]
            if label in numbers_miniwaves:
                cv2.rectangle(img_bb, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(img_bb, str(label), (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, color=(0, 255, 0))
            elif label in numbers_waves:
                cv2.rectangle(img_bb, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(img_bb, str(label), (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, color=(0, 255, 0))
            elif label in numbers_long_sparks:
                cv2.rectangle(img_bb, (x, y), (x + w, y + h), (255, 191, 0), 2)
                cv2.putText(img_bb, str(label), (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, color=(0, 255, 0))
            else:
                cv2.rectangle(img_bb, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(img_bb, str(label), (x, y ), cv2.FONT_HERSHEY_PLAIN, 1.5, color=(0, 255, 0))

        plt.imshow(img_bb)
        plt.title("Bounding boxes of calcium release events")
        plt.grid(False)

        if parameters["img_output_folder"]:
            plt.savefig(os.path.join(parameters["img_output_folder"], "imgBoundingBoxes.png"), bbox_inches='tight')

    t14 = time.time()
    if _TO_PRINT_TIMING:
        print('bbox extraction and regrading = ' + str(t14 - t12))

    ## Plotting density plots
    if parameters["plotting_density_general"]:
        # getting distance to nearest spark
        dist_to_spark = scipy.ndimage.morphology.distance_transform_edt(numbered_mask_after_split == 0)
        f1 = imagesc.plot(dist_to_spark, linewidth=0)
        plt.gcf().set_dpi(parameters["img_DPI"])
        plt.title("Distance to nearest spark")
        plt.grid(False)

        if parameters["img_output_folder"]:
            plt.savefig(os.path.join(parameters["img_output_folder"], "imgDensityGeneral.png"), bbox_inches='tight', dpi=parameters["img_DPI"])

    if parameters["plotting_density_preceding"]:
        # getting distance to preceding spark - within 2 um radius from each pixel.
        dist_to_preceding = np.zeros(numbered_mask_after_split.shape)
        preceding_radius = round(2 / parameters["pixel_width_um"])

        # for each column, take a subimage with +- radius around, taking sum of each row and binarizing versus ==0. Then row==1 <=> there is a precedent for the following pixels, until the next spark is reached
        for i_column in range(0, numbered_mask_after_split.shape[1]):
            col_start = np.max([0, i_column - preceding_radius])
            col_end = np.min([numbered_mask_after_split.shape[1], i_column + preceding_radius])
            trace_is_precedent = np.sum(numbered_mask_after_split[:, col_start:col_end], axis=1) > 0

            is_leading_pre_spark = 1  # we start in state 0 = initial segment before any cells (filling distance to precedent with NaN). We can transition to state 1 (in precedent cell/s). From state 1, we can move to 2 (following a precedent). We don't explicitly keep track of states, this is just the general idea.
            counter_from_precedent = 1  # what we fill to the first after-precedent position
            for i_row in range(len(trace_is_precedent)):
                if trace_is_precedent[i_row] == True:  # if we're in a precedent object
                    dist_to_preceding[i_row, i_column] = 0
                    is_leading_pre_spark = 0
                    counter_from_precedent = 1
                elif is_leading_pre_spark == 1:  # we're before reaching the first object
                    dist_to_preceding[i_row, i_column] = np.nan
                else:  # we're following a precedent object
                    dist_to_preceding[i_row, i_column] = counter_from_precedent
                    counter_from_precedent = counter_from_precedent + 1

        #fig = plt.figure(dpi=parameters["img_DPI"])
        f2 = imagesc.plot(dist_to_preceding, linewidth=0)  
        plt.gcf().set_dpi(parameters["img_DPI"])
        plt.title("Distance to nearest preceding spark")
        plt.grid(False)

        if parameters["img_output_folder"]:
            plt.savefig(os.path.join(parameters["img_output_folder"], "imgDensityPreceding.png"), bbox_inches='tight', dpi=parameters["img_DPI"])

    t15 = time.time()
    if _TO_PRINT_TIMING:
        print('density plots = ' + str(t15 - t14))

    numbered_mask = numbered_mask_after_split
    img_smoothed = img_den_gauss
    return numbered_mask, img_smoothed, img_smoothed_not_rcnormalized, img_SD_multiple, numbers_waves, numbers_miniwaves, numbers_long_sparks


def analyze_sparks(numbered_mask, img_smoothed, numbers_waves, numbers_miniwaves, numbers_long_sparks, parameters):
    """Analyzes features of calcium sparks and other segmented objects.

        Parameters
        ----------
        numbered_mask : numpy array
            a 'numbered mask' describing the segmentation. This is a numpy array with the same size as the input image containing i's in the positions of object with the label i
        img_smoothed : numpy array
            the input image after denoising
        numbers_waves : list
            the list of object labels classified as waves
        numbers_miniwaves : list
            the list of object labels classified as miniwaves
        numbers_long_sparks : list
            the list of object labels classified as long sparks
        parameters : dict
            the dictionary corresponding to a structure of parameters
        Returns
        -------
        spark_feature_matrix : Pandas dataframe
            each row corresponds to features of a single spark
        spark_feature_matrix_summary : Pandas dataframe
            a single-row dataframe with numerical summary (median, lower, and upper quartile) of the columns in spark_feature_matrix
        traces: list
            The list of all traces of single sparks
        """

    props = regionprops(numbered_mask)

    # Now we create a range of vectors which are subsequently filled and stored in a Panda dataframe
    n_features = 20
    n_events = len(props)

    if n_events == 0:
        return {},{},{}

    results_label = [0] * n_events
    results_object_type = ["spark" for i in range(n_events)]  # by default, we define everything as sparks
    results_amplitude = [0] * n_events  #in fluorescence units
    results_amplitudeDFF0 = [0] * n_events
    results_FW_pixels = [0] * n_events
    results_FD_pixels = [0] * n_events  # full duration
    results_FWHM_pixels = [0] * n_events
    results_FDHM_pixels = [0] * n_events  # full duration at half-max amplitude
    results_FW_um = [0] * n_events
    results_FD_ms = [0] * n_events  # full duration
    results_FWHM_um = [0] * n_events
    results_FDHM_ms = [0] * n_events  # full duration at half-max amplitude
    results_BB_min_row = [0] * n_events  # All 0-based
    results_BB_min_col = [0] * n_events  #
    results_BB_max_row = [0] * n_events  #
    results_BB_max_col = [0] * n_events  #
    results_time_to_peak_ms = [0] * n_events
    results_decay_tau_ms = [0] * n_events
    results_duration_to_precedent = [0] * n_events
    results_is_edge = [0] * n_events
    results_spark_frequency = ["" for i in range(n_events)]
    results_long_spark_frequency = ["" for i in range(n_events)]
    results_miniwave_frequency = ["" for i in range(n_events)]
    results_wave_frequency = ["" for i in range(n_events)]

    traces = []
    for i_object in range(n_events):
        #print(i_object)
        spark_props = props[i_object]

        # Label
        results_label[i_object] = spark_props.label

        # Object type classification
        if spark_props.label in numbers_waves:
            results_object_type[i_object] = "wave"
        elif spark_props.label in numbers_miniwaves:
            results_object_type[i_object] = "miniwave"
        elif spark_props.label in numbers_long_sparks:
            results_object_type[i_object] = "long_spark"

        # Bounding box, converted to be from-to, rather than from-(to+1) as it is now
        bb = spark_props.bbox  # (min_row, min_col, max_row, max_col)
        results_BB_min_row[i_object] = bb[0]
        results_BB_min_col[i_object] = bb[1]
        results_BB_max_row[i_object] = bb[2] - 1
        results_BB_max_col[i_object] = bb[3] - 1

        results_is_edge[i_object] = 0
        if bb[0] <= 0:
            results_is_edge[i_object] = -1 # object is present at the start already
        elif bb[2] >= (numbered_mask.shape[0]-1):
            results_is_edge[i_object] = 1  # object is present at the end


        ## Getting the sub-image corresponding to the spark
        add_start = 0
        add_end = round(50/parameters["pixel_duration_ms"])
        start_row = np.max([0, bb[0]-add_start])
        end_row = np.min([numbered_mask.shape[0], bb[2]+add_end])
        sub_image = img_smoothed[start_row:end_row, bb[1]:bb[3]].copy()  # Mind the copy! Apparently, in Python taking a submatrix maintains a link to the original variable, so any setting of sub_image to NaNs also spawns them in img_smoothed.

        # we remove other sparks present in the subimage
        sub_image[np.logical_and((numbered_mask[start_row:end_row, bb[1]:bb[3]] > 0), (numbered_mask[start_row:end_row, bb[1]:bb[3]] != spark_props.label))] = np.nan

        # this may have produced small isolated islands of non-nans - we delete them
        binary_mask = sub_image > 0
        label_mask = measure.label(binary_mask)
        largest_CC = label_mask == np.argmax(np.bincount(label_mask.flat)[1:])+1
        sub_image[largest_CC == False] = np.nan

        # Removing all-nan rows and columns (thanks to the previous step, these have to be leading/trailing, i.e., we're not skipping time/space)
        sub_image = sub_image[~np.isnan(sub_image).all(axis=1), :]
        sub_image = sub_image[:, ~np.isnan(sub_image).all(axis=0)]

        ## Full width and duration
        results_FW_pixels[i_object] = results_BB_max_col[i_object] - results_BB_min_col[i_object]
        results_FD_pixels[i_object] = results_BB_max_row[i_object] - results_BB_min_row[i_object]  # To get this, we don't use the subimage, where we added some pixels at the end potentially - that is used only for trace analysis.
        results_FW_um[i_object] = results_FW_pixels[i_object] * parameters["pixel_width_um"]
        results_FD_ms[i_object] = results_FD_pixels[i_object] * parameters["pixel_duration_ms"]

        # Getting and measuring the mean trace of the subimage.
        trace = np.nanmean(sub_image, axis=1)
        traces.append(trace)
        # trace2 = sub_image[:,round(sub_image.shape[1]/2)] #central column, not used
        # Amplitude
        baseline = (trace[0] + trace[len(trace)-1]) / 2  # We define baseline as the average between first and last element
        peak = np.max(trace)
        results_amplitude[i_object] = peak - baseline

        results_amplitudeDFF0[i_object] = (peak - baseline)/baseline

        # Half-duration
        threshold = baseline + 0.5 * (peak - baseline)
        results_FDHM_pixels[i_object] = longest_seq(trace > threshold, 1)  # we measure the number of lines over threshold. We could probably do linear interpolation, but likely not important...
        results_FDHM_ms[i_object] = parameters["pixel_duration_ms"] * results_FDHM_pixels[i_object]

        # Half-width
        trace_spatial = np.nanmean(sub_image, axis=0) # now we average the spark over time, getting a spatial profile
        baseline_spatial = (trace_spatial[0] + trace_spatial[len(trace_spatial) - 1]) / 2  # We define baseline as the average between first and last element
        peak_spatial = np.max(trace_spatial)
        threshold_spatial = baseline_spatial + 0.5 * (peak_spatial - baseline_spatial)
        results_FWHM_pixels[i_object] = longest_seq(trace_spatial > threshold_spatial, 1)
        results_FWHM_um[i_object] = parameters["pixel_width_um"] * results_FWHM_pixels[i_object]

        # Time to peak
        where_peak = np.argmax(trace)
        results_time_to_peak_ms[i_object] = where_peak

        # decay
        trace_from_peak = trace[where_peak:len(trace)]
        time_from_peak = np.linspace(0, (len(trace_from_peak)-1)*parameters["pixel_duration_ms"], len(trace_from_peak))
        if len(time_from_peak) <= 3:  # If there isn't enough data, no fitting is performed
            results_decay_tau_ms[i_object] = np.nan
        else:
            param_a, param_tau, param_offset = fit_exp_nonlinear(time_from_peak, trace_from_peak)
            results_decay_tau_ms[i_object] = param_tau

        # distance from preceding spark. For our spark, we take the bounding box and go upward, checking 2 um to the sides, until we hit a spark.
        # Then we take the distance
        preceding_radius = round(2 / parameters["pixel_width_um"])
        col_start = np.max([0, results_BB_min_col[i_object] - preceding_radius])
        col_end = np.min([numbered_mask.shape[1], results_BB_max_col[i_object] + preceding_radius])

        trace_is_precedent = np.sum(numbered_mask[0:results_BB_min_row[i_object], col_start:col_end], axis=1) > 0
        where_precedents = np.argwhere(trace_is_precedent==True)
        if len(where_precedents > 0):  # there is a preceding spark - get the delay
            row_precedent = where_precedents[-1][0]
            results_duration_to_precedent[i_object] = parameters["pixel_duration_ms"] * (results_BB_min_row[i_object] - row_precedent - 1)
        else:  # No preceding spark
            results_duration_to_precedent[i_object] = np.nan

    # Calculating frequency of various events
    recording_duration = parameters["pixel_duration_ms"] * numbered_mask.shape[0] / 1000  # Duration of recording in seconds

    n_long_sparks = results_object_type.count("long_spark")
    n_waves = results_object_type.count("wave")
    n_miniwaves = results_object_type.count("miniwave")
    n_sparks = results_object_type.count("spark")

    spark_frequency = n_sparks / recording_duration
    spark_frequency_per100um = spark_frequency / (numbered_mask.shape[1] * parameters["pixel_width_um"]/100)
    long_spark_frequency = n_long_sparks / recording_duration
    long_spark_frequency_per100um = long_spark_frequency / (numbered_mask.shape[1] * parameters["pixel_width_um"] / 100)
    wave_frequency = n_waves / recording_duration
    wave_frequency_per100um = wave_frequency / (numbered_mask.shape[1] * parameters["pixel_width_um"] / 100)
    miniwave_frequency = n_miniwaves / recording_duration
    miniwave_frequency_per100um = miniwave_frequency / (numbered_mask.shape[1] * parameters["pixel_width_um"] / 100)

    results_spark_frequency[0] = str(spark_frequency_per100um)
    results_long_spark_frequency[0] = str(long_spark_frequency_per100um)
    results_wave_frequency[0] = str(wave_frequency_per100um)
    results_miniwave_frequency[0] = str(miniwave_frequency_per100um)

    # saving all vectors in pandas dataframe
    spark_feature_matrix = pd.DataFrame({"object id": results_label, "object type": results_object_type, "amplitude (fluorescence units)": results_amplitude, "amplitude (delta F/F0)": results_amplitudeDFF0, "full width (pixels)": results_FW_pixels, "full duration (pixels)": results_FD_pixels,
                                         "full width at half-max amplitude (pixels)": results_FWHM_pixels, "full duration at half-max amplitude (pixels)": results_FDHM_pixels, "full width (um)": results_FW_um, "full duration (ms)": results_FD_ms,
                                         "full width at half-max amplitude (um)": results_FWHM_um, "full duration at half-max amplitude (ms)": results_FDHM_ms, "bounding box min row (pixels)": results_BB_min_row, "bounding box min column (pixels)": results_BB_min_col,
                                        "bounding box max row (pixels)": results_BB_max_row, "bounding box max column (pixels)": results_BB_max_col, "time to peak (ms)": results_time_to_peak_ms,
                                        "tau of decay (ms)": results_decay_tau_ms, "delay from preceding object (ms)": results_duration_to_precedent, "is incompletely recorded (0 if complete)": results_is_edge, "sparks per second per 100 um": results_spark_frequency,
                                         "long sparks per second per 100 um": results_long_spark_frequency, "waves per second per 100 um": results_wave_frequency, "miniwaves per second per 100 um": results_miniwave_frequency}) #

    spark_feature_matrix_summary = pd.DataFrame( # a summary containing median and 25%- and 75%- quantile
        {"filename": '', # has to be specified by the user, but convenient to have this pre-defined
         "n_sparks": results_object_type.count("spark"),
         "n_long_sparks": results_object_type.count("long_spark"),
         "n_waves": results_object_type.count("wave"),
         "n_miniwaves": results_object_type.count("miniwave"),
         "MED amplitude (normalized fl. units)": np.nanmedian(results_amplitude),
         "Q25 amplitude (normalized fl. units)": np.nanquantile(results_amplitude, 0.25),
         "Q75 amplitude (normalized fl. units)": np.nanquantile(results_amplitude, 0.75),

         "MED amplitude (delta F/F0)": np.nanmedian(results_amplitudeDFF0),
         "Q25 amplitude (delta F/F0)": np.nanquantile(results_amplitudeDFF0, 0.25),
         "Q75 amplitude (delta F/F0)": np.nanquantile(results_amplitudeDFF0, 0.75),

         "MED full width (pixels)": np.nanmedian(results_FW_pixels),
         "Q25 full width (pixels)": np.nanquantile(results_FW_pixels, 0.25),
         "Q75 full width (pixels)": np.nanquantile(results_FW_pixels, 0.75),

         "MED full duration (pixels)": np.nanmedian(results_FD_pixels),
         "Q25 full duration (pixels)": np.nanquantile(results_FD_pixels, 0.25),
         "Q75 full duration (pixels)": np.nanquantile(results_FD_pixels, 0.75),

         "MED full width at half-max amplitude (pixels)": np.nanmedian(results_FWHM_pixels),
         "Q25 full width at half-max amplitude (pixels)": np.nanquantile(results_FWHM_pixels, 0.25),
         "Q75 full width at half-max amplitude (pixels)": np.nanquantile(results_FWHM_pixels, 0.75),

         "MED full duration at half-max amplitude (pixels)": np.nanmedian(results_FDHM_pixels),
         "Q25 full duration at half-max amplitude (pixels)": np.nanquantile(results_FDHM_pixels, 0.25),
         "Q75 full duration at half-max amplitude (pixels)": np.nanquantile(results_FDHM_pixels, 0.75),

         "MED full width (um)": np.nanmedian(results_FW_um),
         "Q25 full width (um)": np.nanquantile(results_FW_um, 0.25),
         "Q75 full width (um)": np.nanquantile(results_FW_um, 0.75),

         "MED full duration (ms)": np.nanmedian(results_FD_ms),
         "Q25 full duration (ms)": np.nanquantile(results_FD_ms, 0.25),
         "Q75 full duration (ms)": np.nanquantile(results_FD_ms, 0.75),

         "MED full width at half-max amplitude (um)": np.nanmedian(results_FWHM_um),
         "Q25 full width at half-max amplitude (um)": np.nanquantile(results_FWHM_um, 0.25),
         "Q75 full width at half-max amplitude (um)": np.nanquantile(results_FWHM_um, 0.75),

         "MED full duration at half-max amplitude (ms)": np.nanmedian(results_FDHM_ms),
         "Q25 full duration at half-max amplitude (ms)": np.nanquantile(results_FDHM_ms, 0.25),
         "Q75 full duration at half-max amplitude (ms)": np.nanquantile(results_FDHM_ms, 0.75),

         "MED time to peak (ms)": np.nanmedian(results_time_to_peak_ms),
         "Q25 time to peak (ms)": np.nanquantile(results_time_to_peak_ms, 0.25),
         "Q75 time to peak (ms)": np.nanquantile(results_time_to_peak_ms, 0.75),

         "MED tau of decay (ms)": np.nanmedian(results_decay_tau_ms),
         "Q25 tau of decay (ms)": np.nanquantile(results_decay_tau_ms, 0.25),
         "Q75 tau of decay (ms)": np.nanquantile(results_decay_tau_ms, 0.75),

         "MED delay from preceding object (ms)": np.nanmedian(results_duration_to_precedent),
         "Q25 delay from preceding object (ms)": np.nanquantile(results_duration_to_precedent, 0.25),
         "Q75 delay from preceding object (ms)": np.nanquantile(results_duration_to_precedent, 0.75),

         "sparks per second per 100 um": spark_frequency_per100um,
         "long sparks per second per 100 um": long_spark_frequency_per100um,
         "waves per second per 100 um": wave_frequency_per100um,
         "miniwaves per second per 100 um": miniwave_frequency_per100um}, index=[0])  #

    return spark_feature_matrix, spark_feature_matrix_summary, traces


def split_sparks_intensity(numbered_mask, img_SD_multiple, splitting_threshold_step, splitting_min_split_depth, splitting_max_threshold,
                           function_subspark_scoring_brightness, function_subspark_scoring_size, subspark_threshold, quantile_level, object_numbers_no_split):
    """ A function for splitting possible spark clusters into single sparks (or at least subclusters)

        Parameters
        ----------
        numbered_mask : numpy array
            a 'numbered mask' describing the segmentation. This is a numpy array with the same size as the input image containing i's in the positions of object with the label i
        img_SD_multiple : numpy array
            the SD-transformed version of the previous output image. This gives, for each pixel, how many standard deviation (across the whole image's pixels) the pixel is from the image's mean intensity.
        splitting_threshold_step : float
            the step size for generation of vector of splitting thresholds to be explored. The smaller this is, the finer-grained the splitting is.
        splitting_min_split_depth : int
            the minimum number of consecutive thresholds that lead to a successful split into >= 2 objects, so that a splitting is performed.
        splitting_max_threshold : float
            the maximum threshold (in SD units) at which cutting into multiple objects is attempted.
        function_subspark_scoring_brightness : function (1 numeric parameter)
            a function returning a 0-to-1 score for calcium subsparks (objects that are tentatively formed when a splitting threshold is applied to an object), based on a numerical summary of their brightness (e.q. 75-percentile etc.)
        function_subspark_scoring_size : function (1 numeric parameter)
            a function returning a 0-to-1 score for calcium subsparks (objects that are tentatively formed when a splitting threshold is applied to an object), based on the number of their pixels
        subspark_threshold : float
            the compound score which is sufficient for considering a subspark "spark-like enough". When trying to split an object according to a threshold, it is recorded whether it was split into at least 2 subobjects with compound score higher than this parameter.
        quantile_level : float
            the quantile at which the brightness is measured
        object_numbers_no_split : list
            a list of object for which splitting is not to be attempted (typically this would be long sparks, waves, and miniwaves)

        Returns
        -------
        numbered_mask_after_split : numpy array
            a numbered mask representing objects following splitting. Objects that were not split keep their original numbering; for objects that are split, one of them reuses the original number, while others are given a new number.
        """

    # A function splitting fused spark clusters throughout the image.
    numbered_mask_after_split = numbered_mask

    object_numbers = np.unique(numbered_mask)
    if len(object_numbers) == 0:
        return numbered_mask_after_split

    if (object_numbers[0] == 0):
        object_numbers = np.delete(object_numbers, 0)

    if len(object_numbers) == 0:
        return numbered_mask_after_split

    n_rows = numbered_mask.shape[0]
    n_cols = numbered_mask.shape[1]

    queue = deque(np.setdiff1d(object_numbers, object_numbers_no_split))  # Queue of objects to be split
    next_free_number = max(object_numbers) + 1

    while len(queue) > 0:
        #print(len(queue))
        i_object = queue.popleft() # number of the currently split object

        # find the pixels
        pixels = np.where(numbered_mask == i_object)
        object_rows = pixels[0]
        object_cols = pixels[1]

        min_row = min(object_rows)
        max_row = max(object_rows)
        min_col = min(object_cols)
        max_col = max(object_cols)

        pixel_values = img_SD_multiple[object_rows, object_cols]

        # Now we generate a new binary mask corresponding to only the currently considered object (and a corresponding _values variable, having image_SD_multiple, but only for this object)
        new_frame = np.zeros(numbered_mask.shape)
        new_frame[object_rows, object_cols] = 1

        new_frame_values = np.zeros(numbered_mask.shape)
        new_frame_values[object_rows, object_cols] = pixel_values

        # Moving to a local window only
        new_frame = new_frame[min_row:(max_row+1), min_col:(max_col+1)]
        new_frame_values = new_frame_values[min_row:(max_row + 1), min_col:(max_col+1)]

        # Getting the list of thresholds to be explored
        min_val = min(pixel_values)
        max_val = min(splitting_max_threshold, max(pixel_values))
        thresholds = list(np.arange(min_val, max_val+1e-12, splitting_threshold_step)) # +1e-12 to make sure the max val can be reached

        n_large_components = np.zeros(len(thresholds))

        # We first do a striding pass through the thresholds
        for i_threshold in range(0, len(thresholds), int(splitting_min_split_depth)):
            threshold = thresholds[i_threshold]
            #props = regionprops(measure.label(new_frame_values > threshold))
            props = regionprops(scipy.ndimage.label(new_frame_values > threshold)[0])
            if len(props) == 1:  # If no splitting happened, no point the single object etc.
                n_large_components[i_threshold] = 0
                continue

            object_scores = np.zeros(len(props))
            for i_subobject in range(len(props)):
                subobject_rows = props[i_subobject].coords[:, 0]
                subobject_cols = props[i_subobject].coords[:, 1]
                size_s = function_subspark_scoring_size(len(subobject_rows))
                if (size_s < subspark_threshold):  # If already the size score is low, we can stop early
                    continue
                br_quantile = np.quantile(new_frame_values[subobject_rows, subobject_cols] - threshold, quantile_level)
                br_s = function_subspark_scoring_brightness(br_quantile)
                object_scores[i_subobject] = br_s * size_s
            n_large_components[i_threshold] = sum(object_scores > subspark_threshold)

        # If we haven't detected anything with split in at least 2 components, we quit, as there is no chance to score a deep-enough split even if all other thresholds led to splits to many objects.
        where_splits = np.argwhere(n_large_components > 1)
        if len(where_splits) == 0:
            continue

        # Otherwise, there are some thresholds to be explored - on both sides from each 1 found.
        to_explore = []
        for i_promising in range(len(where_splits)):
            site_split = where_splits[i_promising][0]
            to_explore.extend(range(site_split-int(splitting_min_split_depth)+1, site_split))
            to_explore.extend(range(site_split+1, site_split + int(splitting_min_split_depth)))
        to_explore = np.array(to_explore)
        to_explore = to_explore[to_explore >= 0] # getting rid of unfeasible indices at the edges
        to_explore = to_explore[to_explore < len(thresholds)]
        to_explore = list(set(to_explore))  # getting rid of duplicate entries

        # All thresholds to be explored are now assessed
        for j_threshold in range(len(to_explore)):
            i_threshold = to_explore[j_threshold]
            threshold = thresholds[i_threshold]
            #props = regionprops(measure.label(new_frame_values > threshold))
            props = regionprops(scipy.ndimage.label(new_frame_values > threshold)[0])
            if len(props) == 1:  # If no splitting happened, no point the single object etc.
                n_large_components[i_threshold] = 0
                continue

            object_scores = np.zeros(len(props))
            for i_subobject in range(len(props)):
                subobject_rows = props[i_subobject].coords[:, 0]
                subobject_cols = props[i_subobject].coords[:, 1]

                size_s = function_subspark_scoring_size(len(subobject_rows))
                if (size_s < subspark_threshold):  # If already the size score is low, we can stop early
                    continue
                br_quantile = np.quantile(new_frame_values[subobject_rows, subobject_cols] - threshold, quantile_level)
                br_s = function_subspark_scoring_brightness(br_quantile)
                object_scores[i_subobject] = br_s * size_s
            n_large_components[i_threshold] = sum(object_scores > subspark_threshold)

            # We try early stopping here - if we have a long-enough chunk of good splits, no need to continue.
            #threshold_props = regionprops(measure.label(np.expand_dims(n_large_components, 1) > 1))  # expand_dims because regionprops requires explicitly 2d structure.
            threshold_props = regionprops(scipy.ndimage.label(np.expand_dims(n_large_components, 1) > 1)[0])  # expand_dims because regionprops requires explicitly 2d structure.
            to_break = False
            for i_chunk in range(len(threshold_props)):
                if threshold_props[i_chunk].coords.shape[0] >= splitting_min_split_depth:
                    to_break = True
                    break
            if to_break:  # If we found a long-enough segment, we can stop now.
                break

        # Based on the vector of possible splits, we find the cleanest (longest) split in n_large_components > 1
        #threshold_props = regionprops(measure.label(np.expand_dims(n_large_components, 1) > 1))
        threshold_props = regionprops(scipy.ndimage.label(np.expand_dims(n_large_components, 1) > 1)[0])
        where_max_size = -1  # where in threshold_props is the longest segment/chunk
        max_size = -1  # and its size
        for i_chunk in range(len(threshold_props)):
            if threshold_props[i_chunk].coords.shape[0] > max_size:
                max_size = threshold_props[i_chunk].coords.shape[0]
                where_max_size = i_chunk

        if max_size < int(splitting_min_split_depth):  # if the deepest split is still too shallow, we do not split the currently considered object
            continue

        # If we got here, we select the threshold for the actual splitting (picking it as the lowest threshold in the chunk of well-splitting thresholds)
        threshold_selected = thresholds[threshold_props[where_max_size].coords[0, 0]]
        sub_object_mask = new_frame_values > threshold_selected
        #s = regionprops(measure.label(sub_object_mask))
        s = regionprops(scipy.ndimage.label(sub_object_mask)[0])

        object_scores = np.zeros((len(s), 1))
        is_first = True
        numbered_mask_after_split[object_rows, object_cols] = 0  # we erase the original mask for this object and will replace it with the new objects

        for i_subobject in range(len(s)):
            subobject_rows = s[i_subobject].coords[:, 0]
            subobject_cols = s[i_subobject].coords[:, 1]
            br_quantile = np.quantile(new_frame_values[subobject_rows, subobject_cols] - threshold_selected, quantile_level)
            br_s = function_subspark_scoring_brightness(br_quantile)
            size_s = function_subspark_scoring_size(s[i_subobject].coords.shape[0])
            object_scores[i_subobject] = br_s * size_s

            if object_scores[i_subobject] <= subspark_threshold:  # if the object is too small, it is discarded
                sub_object_mask[subobject_rows, subobject_cols] = 0
                new_frame_values[subobject_rows, subobject_cols] = 0
                continue

            # If we got here, the object has passed as sufficiently large
            if is_first:  # if we're taking the first subobject, we reuse the original number
                is_first = False
                numbered_mask_after_split[min_row + subobject_rows, min_col + subobject_cols] = i_object
                queue.append(i_object)
            else:  # otherwise we take the next free number
                numbered_mask_after_split[min_row + subobject_rows, min_col + subobject_cols] = next_free_number
                queue.append(next_free_number)
                next_free_number = next_free_number + 1

        # Now, we re-paint the new objects, spreading them to the edges of the low-thresh object
        mask_only_low_thresh_object = np.zeros(sub_object_mask.shape) # This is local-window-sized variable
        nmas_local = numbered_mask_after_split[min_row:(max_row+1), min_col:(max_col+1)]
        mask_only_low_thresh_object[sub_object_mask] = nmas_local[sub_object_mask]
        mask_only_low_thresh_object_binary = mask_only_low_thresh_object > 0

        # For each object in the high mask, we make one "queue" - this will be expanded with pixels that:
        # a) are in the low threshold mask, b) are not yet filled with a particular number, c) are not brighter that currently expanded points
        #props_spreading = regionprops((measure.label(mask_only_low_thresh_object_binary)))
        props_spreading = regionprops(scipy.ndimage.label(mask_only_low_thresh_object_binary)[0])
        number_of_component = np.zeros((len(props_spreading), 1))  # which object number is being spread
        boundaries = []
        for i_boundary in range(len(props_spreading)):
            boundaries.append(props_spreading[i_boundary].coords)
            number_of_component[i_boundary] = int(nmas_local[props_spreading[i_boundary].coords[0, 0], props_spreading[i_boundary].coords[0, 1]])

        # While the queues are not empty, we spread the waves
        while sum([len(boundary) for boundary in boundaries]) > 0:  # while at least one of the boundaries is not empty
            for i_boundary in range(len(props_spreading)):  # for each boundary
                boundary = boundaries[i_boundary]
                new_boundary = []
                for i_point in range(boundary.shape[0]):  # for each point in the boundary
                    point_row = boundary[i_point, 0]
                    point_col = boundary[i_point, 1]
                    neighbours = []
                    if point_row > 0:
                        neighbours.append((point_row - 1, point_col))  #

                    if point_row < nmas_local.shape[0] - 1:
                        neighbours.append((point_row + 1, point_col))

                    if point_col > 0:
                        neighbours.append((point_row, point_col - 1))

                    if point_col < nmas_local.shape[1] - 1:
                        neighbours.append((point_row, point_col + 1))

                    for i_neighbour in range(len(neighbours)):
                        rn = neighbours[i_neighbour][0]
                        rc = neighbours[i_neighbour][1]

                        # We add the point to the new boundary if a) the new pixel is not yet assigned, b) it is within the low-thresh object, c) it is not brighter than the current point.

                        if (nmas_local[rn, rc] == 0) & (new_frame[rn, rc] > 0) & (new_frame_values[point_row, point_col] >= new_frame_values[rn, rc]):
                            nmas_local[rn, rc] = number_of_component[i_boundary]
                            new_boundary.append((rn, rc))

                boundaries[i_boundary] = np.array(new_boundary)

        # And now we move the local numbered mask to the global mask
        numbered_mask_after_split[min_row:(max_row+1), min_col:(max_col+1)] = nmas_local
        #print("ok")

    return numbered_mask_after_split

    #def evaluate_list_of_parameters(list_images, list_reference_images, list_parameters, penalty_fp, penalty_fn):
    #    print('evaluating x')

def measure_map_overlap(mask_reference, mask_predicted, minimum_dice_overlap, penalty_false_positive = 1, penalty_false_negative = 1):
    """ The function takes two segmentation maps (numbered, containing is for the ith object) and counts false x {positives, negatives},
    using the first map as the reference. An overlap can be scored between any object numbers. Maximum bipartite matching approach according to
    the Dice coefficient is used to assign objects in the predicted mask to the reference.

        Parameters
        ----------
        mask_reference : numpy array
            a reference/ground truth 'numbered mask' describing the segmentation. This is a numpy array with the same size as the input image containing i's in the positions of object with the label i.
        mask_predicted : numpy array
            the same type of variable as mask_reference, but this is the predicted mask, which is to be scored versus the ground truth.
        minimum_dice_overlap : float
            minimum Dice coefficient (https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient)  that is sufficient to score as an overlap between an object in the reference versus predicted mask.
        penalty_false_negative : list
            the weight of false negatives when returning the overall overlap error score.
        penalty_false_negative : list
            the weight of false positives in the overall overlap error score.
        Returns
        -------
        out_score : float
            overall score determining the overlap error (i.e., the higher this is, the worse the overlap). It is defined as a weighted average of false negatives and false positives.
        classification : dictionary
            A dictionary counting the true positives, false positives, and false negatives ("TP", "FP", "FN")
        """

    regionprops_reference = regionprops(mask_reference)
    regionprops_predicted = regionprops(mask_predicted)

    # The code below can be uncommented to get a visual illustration of the overlap.
    # map_together = np.zeros((map_reference.shape[0], map_reference.shape[1], 3)) # green for both, blue for reference only, yellow for predicted only
    # for i_row in range(map_reference.shape[0]):
    #     for i_col in range(map_reference.shape[1]):
    #
    #         if (map_reference[i_row, i_col] > 0) & (map_predicted[i_row, i_col] > 0):  # both there -> green
    #             map_together[i_row, i_col, 0] = 0
    #             map_together[i_row, i_col, 1] = 1
    #             map_together[i_row, i_col, 2] = 0
    #         elif (map_reference[i_row, i_col] > 0) & (map_predicted[i_row, i_col] == 0):  # ref only -> blue
    #             map_together[i_row, i_col, 0] = 0
    #             map_together[i_row, i_col, 1] = 0
    #             map_together[i_row, i_col, 2] = 1
    #         elif (map_reference[i_row, i_col] == 0) & (map_predicted[i_row, i_col] > 0):  # pred only -> yellow
    #             map_together[i_row, i_col, 0] = 1
    #             map_together[i_row, i_col, 1] = 1
    #             map_together[i_row, i_col, 2] = 0
    #             # otherwise both background, ignore
    #
    # fig = plt.figure(997)
    # rgb_coloring = numbered_mask_to_RGB(map_reference)
    # plt.imshow(rgb_coloring)
    # plt.title("Map reference")
    # plt.grid(False)
    # fig = plt.figure(998)
    # rgb_coloring2 = numbered_mask_to_RGB(map_predicted)
    # plt.imshow(rgb_coloring2)
    # plt.title("Map predicted")
    # plt.grid(False)
    # fig = plt.figure(999)
    # plt.imshow(map_together)
    # plt.title("Map overlap")
    # plt.grid(False)


    dice_score = np.zeros((len(regionprops_reference), len(regionprops_predicted)))
    for i_reference in range(len(regionprops_reference)):
        aset = set([tuple(x) for x in regionprops_reference[i_reference].coords])
        for i_predicted in range(len(regionprops_predicted)):
            bset = set([tuple(x) for x in regionprops_predicted[i_predicted].coords])
            overlap_size = len( np.array([x for x in aset & bset]))
            #res = sum(sum(a == b for b in regionprops_reference[i_reference].coords) for a in regionprops_predicted[i_predicted].coords)
            dice_score[i_reference, i_predicted] = 2 * overlap_size / (len(aset) + len(bset) )

    adjacency = dice_score > minimum_dice_overlap
    adjacency_sparse = scipy.sparse.csr_matrix(adjacency)

    # this gives, for each object in reference, the index in the set of predicted. -1s are false negatives. >=0 are correctly detected reference objects
    matching_reference = scipy.sparse.csgraph.maximum_bipartite_matching(adjacency_sparse, perm_type='column')

    # counting true/false positives and negatives
    n_false_negatives = sum(matching_reference == -1)
    n_true_positives = sum(matching_reference >= 0)
    n_false_positives = len(regionprops_predicted) - n_true_positives

    # another way of getting false positives, but should be identical...
    # matching_predicted = scipy.sparse.csgraph.maximum_bipartite_matching(adjacency_sparse, perm_type='row')
    #false_positives = sum(matching_predicted == -1)

    out_score = penalty_false_positive * n_false_positives + penalty_false_negative * n_false_negatives
    classification = {}
    classification["TP"] = n_true_positives
    classification["FP"] = n_false_positives
    classification["FN"] = n_false_negatives
    return out_score, classification


def evaluate_list_of_parameters(imgs, imgs_reference, params_to_explore, penalty_fp, penalty_fn, min_dice):
    """ A function carrying out a parameter sweep, calculating segmentation errors for each image (and a provided reference segmentation) and set of parameters
     provided (the total number of segmentations is thus #images x #parameter-sets). This is essentially equivalent to calling SM2.measure_map_overlap for each
     combination of image and parameter set.

        Parameters
        ----------
        imgs : list
            a list of images to be segmented (these should be single-channel numpy arrays)
        imgs_reference : list
            reference segmentation images, one per each member of imgs. Each reference image should be a three-channel (RGB) numpy array, with red blobs (in the 1st channel) corresponding to reference segmented objects. Detection of "red blobs" is done by comparing the red (1st) image channel to the green (2nd) channel, and taking pixels at least 20 points (on the scale of 0-255) more red than green.
        params_to_explore : list
            a list of SM2 parameters. This does not have to have the same number of elements as the lists above.
        minimum_dice_overlap : list
            a list (one per each member of params_to_explore) of minimum Dice coefficient (https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient)  that is sufficient to score as an overlap between an object in the reference versus predicted mask.
        penalty_false_negative : list
            a list (one per each member of params_to_explore)  of the weight of false negatives when returning the overall overlap error score.
        penalty_false_negative : list
            a list (one per each member of params_to_explore) of the weight of false positives in the overall overlap error score.
        Returns
        -------
        error scores : numpy array
            a 2D numpy array, where error_scores[i,j] gives the overall segmentation error score for the i-th image and j-th set of parameters.
        n_fp : numpy array
            a 2D numpy array, where error_scores[i,j] gives the number of false positives for the i-th image and j-th set of parameters.
        n_fp : numpy array
            a 2D numpy array, where error_scores[i,j] gives the number of false negatives for the i-th image and j-th set of parameters.
        """

    error_scores = np.zeros((len(params_to_explore), len(imgs)))
    n_fp = np.zeros((len(params_to_explore), len(imgs)))
    n_fn = np.zeros((len(params_to_explore), len(imgs)))

    for i_image in range(len(imgs)):
        img = imgs[i_image]
        img_reference = imgs_reference[i_image]
        if np.max(img_reference) <= 1: # if image read as 0-1 scaled, it is multiplied so that the detection of places at least 20 units redder than green channel works.
            img_reference = img_reference * 255
        # Generate numbered map from the reference
        reference_binary = img_reference[:, :, 0] > (img_reference[:, :, 1] + 20)
        reference_numbered = measure.label(reference_binary > 0)

        for i_params in range(len(params_to_explore)):

            print("Processing file " + str(i_image + 1) + "/" + str(len(imgs)) + ", parameters " + str(i_params + 1) + "/" + str(len(params_to_explore)))

            # For each threshold to be considered, obtain segmentation using SM2.segment_sparks
            new_params = params_to_explore[i_params]

            mask_predicted_numbered, _, _, _, waves, miniwaves, long_sparks_numbering = segment_sparks(img, new_params)
            # mask_predicted_numbered, image_SD_multiple
            score, classification = measure_map_overlap(reference_numbered, mask_predicted_numbered, min_dice[i_params], penalty_false_positive=penalty_fp[i_params],
                                                            penalty_false_negative=penalty_fn[i_params])

            error_scores[i_params, i_image] = score
            n_fp[i_params, i_image] = classification["FP"]
            n_fn[i_params, i_image] = classification["FN"]

    return n_fp, n_fn, error_scores



def model_func(t, a, tau, offset):
    """A function of exponential decay that is used when estimating decay tau of calcium sparks

        Parameters
        ----------
        t : float
            time point at which the function is evaluated
        a : float
            a parameter controlling the function's maximum (not exclusively though, offset is also used)
        tau : float
            the exponential decay time constant
        offset : float
            the offset of the whole exponential decay over 0

        Returns
        -------
        : float
            the value of the exponential decay function at point t.
        """
    return a * np.exp(-t/tau) + offset


def fit_exp_nonlinear(t, y):
    """A function for fitting an exponential decay curve to data (usually the downstroke of a Ca spark).
       The equation of the decay used is a * np.exp(-t/tau) + offset.

        Parameters
        ----------
        t : numpy array
            time vector for the data
        y : numpy array
            a parameter controlling the function's maximum (not exclusively though, offset is also used)

        Returns
        -------
        a, tau, offset: float, float, float
            parameters of exponential decay function (a * np.exp(-t/tau) + offset) fit to the input data. When fitting fails (or there isn't enough data points), nans are returned.
        """
    try:
        opt_params, parm_cov = scipy.optimize.curve_fit(model_func, t, y, maxfev=10000)
        a, tau, offset = opt_params
        return a, tau, offset
    except ValueError:
        return np.nan, np.nan, np.nan
    except RuntimeError:
        return np.nan, np.nan, np.nan

def vector_mean(x, diameter):
    """A function calculating the local average on an input vector. It handles boundaries by not considering any values outside the input vector, averaging an incomplete window.

            Parameters
            ----------
            x : numpy array
                input vector of numbers.
            diameter : int
                the diameter of the averaging window applied to each point.

            Returns
            -------
            out : numpy array
                the input vector with local averaging applied
            """
    out = np.zeros_like(x, dtype=np.float64)
    dim_len = x.shape[0]
    for i in range(dim_len):
        if diameter%2 == 0:
            a, b = i - (diameter-1)//2, i + (diameter-1)//2 + 2
        else:
            a, b = i - (diameter-1)//2, i + (diameter-1)//2 + 1

        #cap indices to min and max indices
        a = max(0, a)
        b = min(dim_len, b)
        out[i] = np.mean(x[a:b])
    return out


def medfilt2_padded(img, medfiltSize2D, padding_mode):
    """ 2D median filtering using padding around the image to handle boundaries.

            Parameters
            ----------
            img : numpy array
                input image (single-channel)
            medfiltSize2D : (int, int)
                padding & filtering size. The first number corresponds to rows (padding this many rows at the start/end, then averaging this many rows), the second to the columns.
            padding_mode : str
                padding mode passed to the numpy.pad function

            Returns
            -------
            filtered : numpy array
                the input image after 2D median filtering
            """

    # 2d median filtering with padding. In the medfiltSize2D, the first number corresponds to rows (padding this many rows at the start/end, then averaging this many rows), the second to the columns
    img_padded = np.pad(img, ((medfiltSize2D[0],medfiltSize2D[0]), (medfiltSize2D[1],medfiltSize2D[1])), padding_mode)
    filtered = spnd.median_filter(img_padded, medfiltSize2D)

    return filtered[(medfiltSize2D[0]):(medfiltSize2D[0]+img.shape[0]):, (medfiltSize2D[1]):(medfiltSize2D[1]+img.shape[1])]


def numbered_mask_to_RGB(numbered_mask):
    """ A helper function that generates a RGB image color-coding the underlying numbered mask describing release event segmentation. Each object is given a random RGB color.

            Parameters
            ----------
            numbered_mask : numpy array
                a numbered segmentation mask (i's for the ith object)

            Returns
            -------
            random_coloring : numpy array
                a three-channel RGB numpy array representation of the numbered_mask, with each object being painted with a randomly assigned color.
            """

    random_coloring = np.zeros((numbered_mask.shape[0], numbered_mask.shape[1], 3))
    objects_to_pixels = regionprops(numbered_mask)

    for otp in objects_to_pixels:
        coords = otp.coords
        rows = coords[:, 0]
        cols = coords[:, 1]

        random_coloring[rows, cols, 0] = random.uniform(0, 1)
        random_coloring[rows, cols, 1] = random.uniform(0, 1)
        random_coloring[rows, cols, 2] = random.uniform(0, 1)

    return random_coloring


def longest_seq(A, target):
    """ Finding the length of the longest sequence of a certain number in the input vector of numbers.

            Parameters
            ----------
            A : numpy array
                input sequence of numbers.
            target : float
                target value which is searched for within A.

            Returns
            -------
            max_count : int
                the length of the longest sequence of targets in A.
            """
    cnt, max_count = 0, 0 # running count, and max count
    for e in A:
        cnt = cnt + 1 if e == target else 0  # add to or reset running count
        max_count = max(cnt, max_count) # update max count
    return max_count
