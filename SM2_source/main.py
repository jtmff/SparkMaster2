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

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog, QWidget
from gui import *
from simple_param_popup import Dialog
import SM2
import numpy
import matplotlib.pyplot as plt
import pickle
import os.path
import pandas as pd
from pathlib import Path
import copy
import openpyxl # not called directly, but used in Excel export

import numpy as np
from os.path import exists
import gc
# UNCOMMENT FOR MAC COMPILATION
#import matplotlib
#matplotlib.use(‘qtagg’)

plt.style.use('seaborn-whitegrid')

class MyWindow(QMainWindow,Ui_MainWindow):
    """A class providing GUI functionality to access the SM2 functionality."""
    parameters = {}
    segmentation_outputs = {}
    analysis_outputs = {}
    current_image = []
    autofit_image_fnames = []


    def __init__(self, parent=None):
        """Initialize the GUI, setting handling of events via sockets (~listeners)."""
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)

        #self.isLoading = 0
        ## Setting tooltips
        # Basic
        self.label.setToolTip("<html><head/><body><p> The width of a pixel in micrometers – this is something you can get from your microscope.  </p></body></html>")
        self.label_2.setToolTip("<html><head/><body><p> Temporal resolution of the recording in lines per second. </p></body></html>")
        self.label_3.setToolTip("<html><head/><body><p> A threshold on the scale of 0-1, defining the minimum “compound score” for which an object is labelled as a spark (or other calcium release event); objects with a smaller score are discarded. Therefore, the higher the value of this threshold, the fewer sparks will be detected. The compound score integrates information on object size and brightness.  </p></body></html>")
        self.label_basic_parameters_extracellular_space_threshold.setToolTip(
            "<html><head/><body><p>This parameter is useful for scenarios where your recording contains a part of a cell and a part of extracellular space (which tends to be much darker). This threshold (in image brightness units; usually 0-255 or 0-65535, depending on image encoding) is used in a way that columns with mainly pixels valued less than this threshold are cut off and discarded, and the analysis is carried out only on the cellular part. If your recording does not contain any extracellular space, this parameter can be set to 0 to make sure that no cutting is carried out."   "</p></body></html>")

        self.label_7.setToolTip("<html><head/><body><p> the width of the median filtering applied to the image before further processing. The pair of width and duration in pixels corresponds to the size parameter of scipy.ndimage. median_filter  </p></body></html>")
        self.label_11.setToolTip("<html><head/><body><p>  The duration of the median filtering applied to the image before further processing. The pair of width and duration in pixels corresponds to the size parameter of scipy.ndimage. median_filter </p></body></html>")
        self.label_5.setToolTip("<html><head/><body><p> The width of a single-line Gaussian filter (applied in the way that it performs spatial averaging, but not temporal).  </p></body></html>")
        self.label_9.setToolTip("<html><head/><body><p>  The length of an averaging filter used to subtract the temporal trend from the data. This should always be fairly long (at least twice as long as the duration of wave events), so that it does not start picking up normal variation over time (arising from sparks and/or waves being present). </p></body></html>")

        self.label_6.setToolTip("<html><head/><body><p> A threshold used to select candidate objects that are later tested for being calcium sparks based on the spark detection threshold (within Basic parameters). The higher the value, the more standard deviations away from mean of the post-processed image a pixel has to be to be considered a candidate for further processing.  </p></body></html>")
        self.label_8.setToolTip("<html><head/><body><p>  The radius of a disk which is used for morphological opening  applied to the map of pixels over the object detection threshold. This may be used to discard small unpromising objects that receive a poor compound score later, only slowing the computation.  </p></body></html>")
        self.label_10.setToolTip("<html><head/><body><p> Determines the quantile level at which the brightness of objects is assessed. This parameter is unlikely to be needed to be changed; brightness scoring behaviour is better controlled using the tab “Scoring functions” and the parameters defined there.  </p></body></html>")

        self.label_16.setToolTip("<html><head/><body><p> When generating a list of thresholds to be explored when trying to split an object (a potential spark cluster), this is the step size with which the thresholds are generated. The finer the step, the more cuts are attempted (increasing the likelihood of splitting a spark, but increasing the computation time).   </p></body></html>")
        self.label_17.setToolTip("<html><head/><body><p> How many consecutive thresholds must lead to a split of a spark into at least 2 objects, so that a split is actually performed. When this parameter is too low, it increases the likelihood of spurious spark splitting (oversegmentation). Requesting this parameter to be at least 2-3 considerably reduces the likelihood of splitting of a genuine spark due to chance/presence of noise.  </p></body></html>")
        self.label_18.setToolTip("<html><head/><body><p> The maximum threshold that can be used for exploration of consecutive thresholds. With a high value, splitting of even very bright sparks is attempted, but more thresholds need to be explored, increasing the computation time. </p></body></html>")

        self.ap_long_spark_checkbox_search.setToolTip("<html><head/><body><p>  If checked, objects can be classified as long sparks. This is best left unchecked if you know that your recordings do not contain long sparks. It will make the analysis run slightly faster and without the risk of falsely positive identification of a spark as a long spark.  </p></body></html>")
        self.label_13.setToolTip("<html><head/><body><p> The minimum duration of a spark (or a spark cluster) if it is to be considered a potential long spark and processed further.  </p></body></html>")
        self.label_14.setToolTip("<html><head/><body><p>  Set this to the average width of a long spark in your dataset (it may be the most practical to find the width in pixels and find which value of this parameter in um gives you the appropriate pixel count). </p></body></html>")
        self.label_12.setToolTip("<html><head/><body><p> Sparks wider than this may not be considered long sparks (being probably merely long-lasting spark clusters).  </p></body></html>")
        self.label_15.setToolTip("<html><head/><body><p>  The long sparkiness (see Graphical illustration of SM2 methodology for a technical explanation) is used to separate long sparks from long-lasting clusters of single sparks. Increasing this parameter is one of the ways how false splitting of spark clusters can be avoided.  </p></body></html>")

        self.ap_wave_classification_checkbox_search.setToolTip("<html><head/><body><p> If checked, objects can be classified as waves and/or miniwaves. If you know a priori your data do not contain such objects, it is best to uncheck this. </p></body></html>")
        self.label_19.setToolTip("<html><head/><body><p> This is used in splitting waves and/or miniwaves (which is handled differently from splitting sparks). Set it so that separate sub-objects of wave clusters just about reach over this threshold (in the SD-transformed image, where each pixel is replaced by how many standard deviations away it is from mean; both calculated on the denoised image).  </p></body></html>")
        self.label_20.setToolTip("<html><head/><body><p> The lower this parameter, the weaker-scoring objects (using the wave scoring functions) may be considered to be miniwaves. If, on the other hand, regular sparks are labelled as miniwaves (being shown in cyan blue frames), increasing this parameter (or changing the wave scoring functions) can be used to avoid this.  </p></body></html>")
        self.label_21.setToolTip("<html><head/><body><p>  Similar to the parameter above, except for waves. This parameter should always be at least as high as the miniwave score threshold. </p></body></html>")


        self.label_22.setToolTip("<html><head/><body><p> Parameters of five object-scoring sigmoids are defined here. For each sigmoid, the “Plot” button enables you to visualize the corresponding curve. Each of these sigmoids is scaled between 0 and 1. The “midpoint” is the value at which a sigmoid takes the value of 0.5. The sigmoid slope defines how steeply increasing the sigmoid is around the midpoint.  </p></body></html>")
        self.label_23.setToolTip("<html><head/><body><p> Parameters of five object-scoring sigmoids are defined here. For each sigmoid, the “Plot” button enables you to visualize the corresponding curve. Each of these sigmoids is scaled between 0 and 1. The “midpoint” is the value at which a sigmoid takes the value of 0.5. The sigmoid slope defines how steeply increasing the sigmoid is around the midpoint.  </p></body></html>")
        self.label_30.setToolTip("<html><head/><body><p> Parameters of five object-scoring sigmoids are defined here. For each sigmoid, the “Plot” button enables you to visualize the corresponding curve. Each of these sigmoids is scaled between 0 and 1. The “midpoint” is the value at which a sigmoid takes the value of 0.5. The sigmoid slope defines how steeply increasing the sigmoid is around the midpoint.   </p></body></html>")
        self.label_29.setToolTip("<html><head/><body><p> Parameters of five object-scoring sigmoids are defined here. For each sigmoid, the “Plot” button enables you to visualize the corresponding curve. Each of these sigmoids is scaled between 0 and 1. The “midpoint” is the value at which a sigmoid takes the value of 0.5. The sigmoid slope defines how steeply increasing the sigmoid is around the midpoint.  </p></body></html>")
        self.label_26.setToolTip("<html><head/><body><p> Parameters of five object-scoring sigmoids are defined here. For each sigmoid, the “Plot” button enables you to visualize the corresponding curve. Each of these sigmoids is scaled between 0 and 1. The “midpoint” is the value at which a sigmoid takes the value of 0.5. The sigmoid slope defines how steeply increasing the sigmoid is around the midpoint.  </p></body></html>")
        self.label_28.setToolTip("<html><head/><body><p>  Parameters of five object-scoring sigmoids are defined here. For each sigmoid, the “Plot” button enables you to visualize the corresponding curve. Each of these sigmoids is scaled between 0 and 1. The “midpoint” is the value at which a sigmoid takes the value of 0.5. The sigmoid slope defines how steeply increasing the sigmoid is around the midpoint. </p></body></html>")
        self.label_34.setToolTip("<html><head/><body><p> Parameters of five object-scoring sigmoids are defined here. For each sigmoid, the “Plot” button enables you to visualize the corresponding curve. Each of these sigmoids is scaled between 0 and 1. The “midpoint” is the value at which a sigmoid takes the value of 0.5. The sigmoid slope defines how steeply increasing the sigmoid is around the midpoint.  </p></body></html>")
        self.label_23.setToolTip("<html><head/><body><p> Parameters of five object-scoring sigmoids are defined here. For each sigmoid, the “Plot” button enables you to visualize the corresponding curve. Each of these sigmoids is scaled between 0 and 1. The “midpoint” is the value at which a sigmoid takes the value of 0.5. The sigmoid slope defines how steeply increasing the sigmoid is around the midpoint.  </p></body></html>")
        self.label_35.setToolTip("<html><head/><body><p> Parameters of five object-scoring sigmoids are defined here. For each sigmoid, the “Plot” button enables you to visualize the corresponding curve. Each of these sigmoids is scaled between 0 and 1. The “midpoint” is the value at which a sigmoid takes the value of 0.5. The sigmoid slope defines how steeply increasing the sigmoid is around the midpoint.  </p></body></html>")
        self.label_22.setToolTip("<html><head/><body><p>  Parameters of five object-scoring sigmoids are defined here. For each sigmoid, the “Plot” button enables you to visualize the corresponding curve. Each of these sigmoids is scaled between 0 and 1. The “midpoint” is the value at which a sigmoid takes the value of 0.5. The sigmoid slope defines how steeply increasing the sigmoid is around the midpoint. </p></body></html>")

        self.ap_plotting_checkbox_rawImg.setToolTip("<html><head/><body><p>  Raw image before filtering.  </p></body></html>")
        self.ap_plotting_checkbox_imgColNormalized.setToolTip("<html><head/><body><p> Raw image after column normalization (to get rid of spatial variability of background intensity, manifesting as bright or dark bands in the data).  </p></body></html>")
        self.ap_plotting_checkbox_imgNormalized.setToolTip("<html><head/><body><p> The image after column normalization, with additional row normalization applied (used to get rid of long-term trends in the data).   </p></body></html>")
        self.ap_plotting_checkbox_imgNormalizedAndSmoothed.setToolTip("<html><head/><body><p> Row- and column-normalized image with median filtering and 1D gaussian filtering (both defined by parameters in the “Image preprocessing tab”) applied – useful for setting denoising parameters.  </p></body></html>")
        self.ap_plotting_checkbox_imgSDtransform.setToolTip("<html><head/><body><p> A transformation of the smoothed image, where each pixel is replaced by the number of standard deviations it is away from the image mean. </p></body></html>")

        self.ap_plotting_checkbox_imgCandidateObjects.setToolTip("<html><head/><body><p> A binary mask of objects over the object detection threshold (defined in Spark detection tab). The threshold is applied to the SD-transformed image. I.e., if your analysis does not label clear sparks as sparks, do make sure they are selected as candidate sparks in this step in the first place – if not, lower the object detection threshold.  </p></body></html>")
        self.ap_plotting_checkbox_imgSizeRawMap.setToolTip("<html><head/><body><p>  A heatmap with each object being color-coded by the number of pixels it covers (useful for changing parameters of scoring sigmoids, in case they do not work ideally). </p></body></html>")
        self.ap_plotting_checkbox_imgSizeScoreMap.setToolTip("<html><head/><body><p> A heatmap color coding the size score of each object (after sigmoid scoring).  </p></body></html>")
        self.ap_plotting_checkbox_imgBrightnessQuantileMap.setToolTip("<html><head/><body><p> Similar to the brightness raw map, but showing the quantile brightness of each object.  </p></body></html>")
        self.ap_plotting_checkbox_imgBrightnessScoreMap.setToolTip("<html><head/><body><p> Similar to the brightness score map, showing the score of the brightness of each object.  </p></body></html>")
        self.ap_plotting_checkbox_imgObjectScoreBeforeSplitting.setToolTip("<html><head/><body><p>  Compound scores of objects before spark splitting is applied (the compound score is the product of size score and brightness score). </p></body></html>")

        self.ap_plotting_checkbox_imgObjectColoringBeforeSplitting.setToolTip("<html><head/><body><p> An image with each detected object being color-coded by a random color.  </p></body></html>")
        self.ap_plotting_checkbox_imgObjectsHighThresholdWaveSplitting.setToolTip("<html><head/><body><p> Similar to “Candidate spark objects” image, but using a higher threshold (wave sub-core threshold, defined in the Wave classification tab). This is best used to set the wave sub-core threshold so that it splits subobjects that are a part of a wave or miniwave into distinct objects.  </p></body></html>")
        self.ap_plotting_checkbox_imgObjectColoringAfterSplitting.setToolTip("<html><head/><body><p> Random color-coding of distinct detected objects after splitting of waves and/or sparks is attempted to segment object clusters.  </p></body></html>")

        self.ap_plotting_checkbox_imgBoundingBoxes.setToolTip("<html><head/><body><p> Bounding boxes of the detected objects. Red corresponds to sparks, orange to long sparks, cyan to miniwaves, and blue to waves.  </p></body></html>")
        self.ap_plotting_checkbox_imgDensityPlotGeneral.setToolTip("<html><head/><body><p> A density map, with the value of each pixel corresponding to the shortest distance (in pixel) corresponding to the distance to the closest object. This visualization can take several seconds per image to be calculated and shown.  </p></body></html>")
        self.ap_plotting_checkbox_imgDensityPlotPreceding.setToolTip("<html><head/><body><p> Another density map, this time the distance to the nearest object that is “preceding” – within 1 um of the current pixel, and occurring before the current pixel’s time point. This visualization can be used to indicate refractoriness, and one can see in certain recordings that miniwaves or waves are followed by long periods of no activation, indicating the sarcoplasmic reticulum release being locally refractory.  </p></body></html>")

        self.ap_image_preprocessing_tickboxSpaceSubtraction.setToolTip("<html><head/><body><p> If ticked, space-wise (across columns) background subtraction is carried out. This can be used to eliminate bright bands in the background, for example. </p></body></html>")
        self.ap_image_preprocessing_tickboxTimeSubtraction.setToolTip(
            "<html><head/><body><p> If ticked, time-wise (across rows) background subtraction is carried out. This can be used to get rid of e.g. background bleaching over time, or other temporal trends. </p></body></html>")

        self.setWindowTitle("SM2")
        ## Loading initial parameters
        default_pixel_width = 0.12
        default_lps = 500

        default_parameters = SM2.get_default_parameters(default_pixel_width, default_lps)
        default_parameters["pixel_width_um"] = default_pixel_width
        default_parameters["lps"] = default_lps

        ## Setting up listeners for change lineEdit.
        self.lineEdit_parameters_basic_pixel_width.editingFinished.connect(self.listener_text_or_tick_changed)
        self.lineEdit_parameters_basic_fps.editingFinished.connect(self.listener_text_or_tick_changed)
        self.lineEdit_parameters_spark_detection_threshold.editingFinished.connect(self.listener_text_or_tick_changed)
        self.lineEdit_parameters_extracellular_space_threshold.editingFinished.connect(self.listener_text_or_tick_changed)

        #self.ap_image_preprocessing_lineEdit_spark_radius.editingFinished.connect(self.listener_textbox_changed)
        self.ap_image_preprocessing_lineEdit_1dGaussFilter.editingFinished.connect(self.listener_text_or_tick_changed)
        self.ap_image_preprocessing_lineEdit_median_width.editingFinished.connect(self.listener_text_or_tick_changed)
        self.ap_image_preprocessing_lineEdit_median_duration.editingFinished.connect(self.listener_text_or_tick_changed)
        self.ap_image_preprocessing_lineEdit_temporal_filter_duration.editingFinished.connect(self.listener_text_or_tick_changed)

        self.ap_spark_detection_lineEdit_object_threshold.editingFinished.connect(self.listener_text_or_tick_changed)
        self.ap_spark_detection_lineEdit_morphological_radius.editingFinished.connect(self.listener_text_or_tick_changed)
        self.ap_spark_detection_lineEdit_brightness_quantile.editingFinished.connect(self.listener_text_or_tick_changed)
        #self.ap_spark_detection_lineEdit_minimum_pixel_density.editingFinished.connect(self.listener_text_or_tick_changed)

        self.ap_long_sparks_lineEdit_long_spark_core_diameter.editingFinished.connect(self.listener_text_or_tick_changed)
        self.ap_long_sparks_lineEdit_minimum_duration.editingFinished.connect(self.listener_text_or_tick_changed)
        self.ap_long_sparks_lineEdit_long_spark_width.editingFinished.connect(self.listener_text_or_tick_changed)
        self.ap_long_sparks_lineEdit_minimum_sparkiness.editingFinished.connect(self.listener_text_or_tick_changed)

        self.ap_spark_splitting_lineEdit_threshold_step.editingFinished.connect(self.listener_text_or_tick_changed)
        self.ap_spark_splitting_lineEdit_minimum_depth.editingFinished.connect(self.listener_text_or_tick_changed)
        self.ap_spark_splitting_lineEdit_maximum_threshold.editingFinished.connect(self.listener_text_or_tick_changed)

        self.ap_wave_classification_lineEdit_wave_subcore_threshold.editingFinished.connect(self.listener_text_or_tick_changed)
        self.ap_wave_classification_lineEdit_miniwave_score_threshold.editingFinished.connect(self.listener_text_or_tick_changed)
        self.ap_wave_classification_lineEdit_wave_score_threshold.editingFinished.connect(self.listener_text_or_tick_changed)

        self.ap_scoring_lineEdit_spark_size_midpoint.editingFinished.connect(self.listener_text_or_tick_changed)
        self.ap_scoring_lineEdit_spark_size_slope.editingFinished.connect(self.listener_text_or_tick_changed)
        self.ap_scoring_lineEdit_spark_brightness_midpoint.editingFinished.connect(self.listener_text_or_tick_changed)
        self.ap_scoring_lineEdit_spark_brightness_slope.editingFinished.connect(self.listener_text_or_tick_changed)
        self.ap_scoring_lineEdit_subspark_size_midpoint.editingFinished.connect(self.listener_text_or_tick_changed)
        self.ap_scoring_lineEdit_subspark_size_slope.editingFinished.connect(self.listener_text_or_tick_changed)
        self.ap_scoring_lineEdit_wave_size_midpoint.editingFinished.connect(self.listener_text_or_tick_changed)
        self.ap_scoring_lineEdit_wave_size_slope.editingFinished.connect(self.listener_text_or_tick_changed)
        self.ap_scoring_lineEdit_wave_brightness_midpoint.editingFinished.connect(self.listener_text_or_tick_changed)
        self.ap_scoring_lineEdit_wave_brightness_slope.editingFinished.connect(self.listener_text_or_tick_changed)

        # Listeners for checkboxes
        self.ap_image_preprocessing_tickboxTimeSubtraction.stateChanged.connect(self.listener_text_or_tick_changed)
        self.ap_image_preprocessing_tickboxSpaceSubtraction.stateChanged.connect(self.listener_text_or_tick_changed)
        self.ap_long_spark_checkbox_search.stateChanged.connect(self.listener_text_or_tick_changed)
        self.ap_wave_classification_checkbox_search.stateChanged.connect(self.listener_text_or_tick_changed)
        self.ap_plotting_checkbox_rawImg.stateChanged.connect(self.listener_text_or_tick_changed)
        self.ap_plotting_checkbox_imgColNormalized.stateChanged.connect(self.listener_text_or_tick_changed)
        self.ap_plotting_checkbox_imgNormalized.stateChanged.connect(self.listener_text_or_tick_changed)
        self.ap_plotting_checkbox_imgNormalizedAndSmoothed.stateChanged.connect(self.listener_text_or_tick_changed)
        self.ap_plotting_checkbox_imgSDtransform.stateChanged.connect(self.listener_text_or_tick_changed)
        self.ap_plotting_checkbox_imgCandidateObjects.stateChanged.connect(self.listener_text_or_tick_changed)
        self.ap_plotting_checkbox_imgSizeRawMap.stateChanged.connect(self.listener_text_or_tick_changed)
        self.ap_plotting_checkbox_imgSizeScoreMap.stateChanged.connect(self.listener_text_or_tick_changed)
        self.ap_plotting_checkbox_imgBrightnessQuantileMap.stateChanged.connect(self.listener_text_or_tick_changed)
        self.ap_plotting_checkbox_imgBrightnessScoreMap.stateChanged.connect(self.listener_text_or_tick_changed)
        self.ap_plotting_checkbox_imgObjectScoreBeforeSplitting.stateChanged.connect(self.listener_text_or_tick_changed)
        self.ap_plotting_checkbox_imgObjectColoringBeforeSplitting.stateChanged.connect(self.listener_text_or_tick_changed)
        self.ap_plotting_checkbox_imgObjectsHighThresholdWaveSplitting.stateChanged.connect(self.listener_text_or_tick_changed)
        self.ap_plotting_checkbox_imgObjectColoringAfterSplitting.stateChanged.connect(self.listener_text_or_tick_changed)
        self.ap_plotting_checkbox_imgBoundingBoxes.stateChanged.connect(self.listener_text_or_tick_changed)
        self.ap_plotting_checkbox_imgDensityPlotGeneral.stateChanged.connect(self.listener_text_or_tick_changed)
        self.ap_plotting_checkbox_imgDensityPlotPreceding.stateChanged.connect(self.listener_text_or_tick_changed)

        # Listeners for sigmoid-plotting buttons
        self.ap_scoring_functions_buttonPlot1.clicked.connect(lambda x: self.listener_sigmoid_button_clicked(1))
        self.ap_scoring_functions_buttonPlot2.clicked.connect(lambda x: self.listener_sigmoid_button_clicked(2))
        self.ap_scoring_functions_buttonPlot3.clicked.connect(lambda x: self.listener_sigmoid_button_clicked(3))
        self.ap_scoring_functions_buttonPlot4.clicked.connect(lambda x: self.listener_sigmoid_button_clicked(4))
        self.ap_scoring_functions_buttonPlot5.clicked.connect(lambda x: self.listener_sigmoid_button_clicked(5))

        # Listener for button browsing for folder where to store images, also a listener for DPI resolution
        self.ap_plotting_button_browse.clicked.connect(self.listener_browse_button_img_saving_folder_clicked)
        self.ap_plotting_lineEdit_DPI.editingFinished.connect(self.listener_text_or_tick_changed)

        # Listener for lineEdit with folder where to store images
        self.ap_plotting_lineEdit_folder_save.editingFinished.connect(self.listener_lineEdit_folder_save_edited)

        # Load image listener
        self.button_load_image.clicked.connect(self.listener_button_load_image)

        # Analyze listener
        self.button_analyze_data.clicked.connect(self.listener_button_analyze_data)

        # Save analysis outputs
        self.button_save_outputs.clicked.connect(self.listener_button_save_outputs)

        # Autofit listener
        self.button_autofit.clicked.connect(self.listener_button_autofit)

        # Batch analysis listener
        self.button_batch_analysis.clicked.connect(self.listener_button_batch_analysis)

        # Save&load parameters listener
        self.button_save_parameters.clicked.connect(self.listener_button_save_parameters)
        self.button_load_parameters.clicked.connect(self.listener_button_load_parameters)

        # Close all figures listener
        self.pushButton_close_all_figures.clicked.connect(self.listener_button_close_all_figures)

        self.dialog = Dialog()
        self.dialog.setupUi(self.dialog, self)

        # saving parameters and updating GUI
        self.parameters = default_parameters
        # update GUI based on parameter values
        self.is_initializing = 1 # when this is 1, we do not do anything in text_or_tick_changed callback - no point in responding to initializing changes, and it did cause issues when some of the parameters were yet undefined.
        self.update_gui_from_parameters()
        self.is_initializing = 0


    def closeEvent(self, event):
        """Overrides the default closing action, also closing all figures."""
        self.listener_button_close_all_figures()

        # close window
        event.accept()
        sys.exit()


    def listener_text_or_tick_changed(self):
        """This socket is triggered when parameters change (either in a textbox, or when a checkbox is ticked/unticked.

        The general logic is that parameters are first updated according to the current values in the gui, and subsequently,
        gui is again updated based on the parameters. This is important e.g. when a spatial/temporal resolution is changed
        in the GUI and all labels that depend on the conversion between pixels and spatial/temporal units need to be rewritten."""
        if (self.is_initializing == 0):
            print("Something changed")
            # 1) Get all values in textboxes, store in parameters
            try:
                self.update_parameters_from_gui()
            except ValueError as inst:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)

                msg.setText("Error in reading parameter values")
                msg.setInformativeText("All processing parameters must be numeric")
                msg.setWindowTitle("Error")
                msg.setDetailedText(str(inst))
                msg.exec_()

            print("parameters from gui updated")
            # 2) Set all GUI elements based on the updated parameters
            self.update_gui_from_parameters()
            print("gui updated from parameters")

    def update_pixel_labels(self):
        """A function updating all the labels showing how spatial/temporal values of parameters translate into pixel distances."""
        # A function that sets label text values according to _pixel versions of parameters.
        # Before calling, do first update parameters to make sure that um/ms-based values are consistent with the pixel values.
        # Nothing is returned, parameters are updated in-place.
        self.ap_image_preprocessing_label_1dGaussFilter_pixels.setText("(= " + str(self.parameters["preprocessing_1dGaussFilter_pixels"]) + " pixels)")
        self.ap_image_preprocessing_label_median_filtering_pixels.setText("(= " + str(self.parameters["preprocessing_medfilt_width_pixels"]) + " pixels)")
        self.ap_image_preprocessing_label_median_duration_pixels.setText("(= " + str(self.parameters["preprocessing_medfilt_duration_pixels"]) + " pixels)")
        self.ap_image_preprocessing_label_temporal_filtering_pixels.setText("(= " + str(self.parameters["preprocessing_row_normalization_filter_length_pixels"]) + " pixels)")

        self.ap_spark_detection_label_morphological_radius_pixels.setText("(= " + str(self.parameters["spark_detection_morpho_radius_pixels"]) + " pixels)")

        self.ap_long_sparks_label_core_diameter_pixels.setText("(= " + str(self.parameters["long_sparks_core_diameter_pixels"]) + " pixels)")
        self.ap_long_sparks_minimum_duration_pixels.setText("(= " + str(self.parameters["long_sparks_min_long_spark_duration_pixels"]) + " pixels)")
        self.ap_long_sparks_label_maximum_width_pixels.setText("(= " + str(self.parameters["long_sparks_max_long_spark_width_pixels"]) + " pixels)")

        self.ap_scoring_label_spark_size_midpoint_pixels.setText("(= " + str(self.parameters["scoring_spark_scoring_size_params_ab"][0]) + " pixels)")
        self.ap_scoring_label_subspark_size_midpoint_pixels.setText("(= " + str(self.parameters["scoring_subspark_scoring_size_params_ab"][0]) + " pixels)")
        self.ap_scoring_label_wave_size_midpoint_pixels.setText("(= " + str(self.parameters["scoring_wave_scoring_size_params_ab"][0]) + " pixels)")

    def update_parameters_from_gui(self):
        """Extract parameter values from the GUI and store them in self.parameters."""
        self.parameters["pixel_width_um"] = float(self.lineEdit_parameters_basic_pixel_width.text())
        self.parameters["lps"] = float(self.lineEdit_parameters_basic_fps.text())

        pixel_duration_ms = 1000 / self.parameters["lps"]
        self.parameters["pixel_duration_ms"] = pixel_duration_ms

        self.parameters["spark_detection_compound_threshold"] = float(self.lineEdit_parameters_spark_detection_threshold.text())
        self.parameters["extracellular_rejection"] = float(self.lineEdit_parameters_extracellular_space_threshold.text())

        # Image preprocessing
        self.parameters["preprocessing_space_background_subtraction"] = bool(self.ap_image_preprocessing_tickboxSpaceSubtraction.isChecked())
        self.parameters["preprocessing_time_background_subtraction"] = bool(self.ap_image_preprocessing_tickboxTimeSubtraction.isChecked())
        self.parameters["preprocessing_1dGaussFilter_um"] = float(self.ap_image_preprocessing_lineEdit_1dGaussFilter.text())
        self.parameters["preprocessing_medfilt_width_um"] = float(self.ap_image_preprocessing_lineEdit_median_width.text())
        self.parameters["preprocessing_medfilt_duration_ms"] = float(self.ap_image_preprocessing_lineEdit_median_duration.text())
        self.parameters["preprocessing_temporal_filter_duration"] = float(self.ap_image_preprocessing_lineEdit_temporal_filter_duration.text())

        # Spark detection
        self.parameters["spark_detection_object_detection_threshold"] = float(self.ap_spark_detection_lineEdit_object_threshold.text())
        self.parameters["spark_detection_morpho_radius_um"] = float(self.ap_spark_detection_lineEdit_morphological_radius.text())
        self.parameters["spark_detection_quantile_level"] = float(self.ap_spark_detection_lineEdit_brightness_quantile.text())
        #self.parameters["spark_detection_min_object_density"] = float(self.ap_spark_detection_lineEdit_minimum_pixel_density.text())

        # Long sparks
        self.parameters["long_sparks_search_for_long_sparks"] = bool(self.ap_long_spark_checkbox_search.isChecked())
        self.parameters["long_sparks_core_diameter_um"] = float(self.ap_long_sparks_lineEdit_long_spark_core_diameter.text())
        self.parameters["long_sparks_min_long_spark_duration_ms"] = float(self.ap_long_sparks_lineEdit_minimum_duration.text())
        self.parameters["long_sparks_max_long_spark_width_um"] = float(self.ap_long_sparks_lineEdit_long_spark_width.text())
        self.parameters["long_sparks_threshold_long_sparkiness"] = float(self.ap_long_sparks_lineEdit_minimum_sparkiness.text())

        # Splitting sparks
        self.parameters["splitting_sparks_splitting_threshold_step"] = float(self.ap_spark_splitting_lineEdit_threshold_step.text())
        self.parameters["splitting_sparks_splitting_min_split_depth"] = float(self.ap_spark_splitting_lineEdit_minimum_depth.text())
        self.parameters["splitting_sparks_splitting_max_threshold"] = float(self.ap_spark_splitting_lineEdit_maximum_threshold.text())

        # Wave classification
        self.parameters["wave_classification_search_for_waves"] = bool(self.ap_wave_classification_checkbox_search.isChecked())
        self.parameters["wave_classification_wave_subcore_detection_threshold"] = float(self.ap_wave_classification_lineEdit_wave_subcore_threshold.text())
        self.parameters["wave_classification_miniwave_threshold"] = float(self.ap_wave_classification_lineEdit_miniwave_score_threshold.text())
        self.parameters["wave_classification_wave_threshold"] = float(self.ap_wave_classification_lineEdit_wave_score_threshold.text())

        # Scoring functions
        self.parameters["scoring_spark_scoring_size_midpoint_umms"] = float(self.ap_scoring_lineEdit_spark_size_midpoint.text())
        self.parameters["scoring_spark_scoring_size_params_ab"][1] = float(self.ap_scoring_lineEdit_spark_size_slope.text())
        self.parameters["scoring_spark_scoring_brightness_params_ab"][0] = float(self.ap_scoring_lineEdit_spark_brightness_midpoint.text())
        self.parameters["scoring_spark_scoring_brightness_params_ab"][1] = float(self.ap_scoring_lineEdit_spark_brightness_slope.text())
        self.parameters["scoring_subspark_scoring_size_midpoint_umms"] = float(self.ap_scoring_lineEdit_subspark_size_midpoint.text())

        self.parameters["scoring_subspark_scoring_size_params_ab"][1] = float(self.ap_scoring_lineEdit_subspark_size_slope.text())
        self.parameters["scoring_wave_scoring_size_midpoint_umms"] = float(self.ap_scoring_lineEdit_wave_size_midpoint.text())
        self.parameters["scoring_wave_scoring_size_params_ab"][1] = float(self.ap_scoring_lineEdit_wave_size_slope.text())
        self.parameters["scoring_wave_scoring_brightness_params_ab"][0] = float(self.ap_scoring_lineEdit_wave_brightness_midpoint.text())
        self.parameters["scoring_wave_scoring_brightness_params_ab"][1] = float(self.ap_scoring_lineEdit_wave_brightness_slope.text())

        self.parameters["plotting_raw_img"] = bool(self.ap_plotting_checkbox_rawImg.isChecked())
        self.parameters["plotting_img_col_normalized"] = bool(self.ap_plotting_checkbox_imgColNormalized.isChecked())
        self.parameters["plotting_img_normalized"] = bool(self.ap_plotting_checkbox_imgNormalized.isChecked())
        self.parameters["plotting_img_normalized_and_smoothed"] = bool(self.ap_plotting_checkbox_imgNormalizedAndSmoothed.isChecked())
        self.parameters["plotting_img_SD_transform"] = bool(self.ap_plotting_checkbox_imgSDtransform.isChecked())
        self.parameters["plotting_img_candidate_objects"] = bool(self.ap_plotting_checkbox_imgCandidateObjects.isChecked())
        self.parameters["plotting_img_size_raw_map"] = bool(self.ap_plotting_checkbox_imgSizeRawMap.isChecked())
        self.parameters["plotting_img_size_score_map"] = bool(self.ap_plotting_checkbox_imgSizeScoreMap.isChecked())
        self.parameters["plotting_img_brightness_quantile_map"] = bool(self.ap_plotting_checkbox_imgBrightnessQuantileMap.isChecked())
        self.parameters["plotting_img_brightness_score_map"] = bool(self.ap_plotting_checkbox_imgBrightnessScoreMap.isChecked())
        self.parameters["plotting_img_object_score_before_splitting"] = bool(self.ap_plotting_checkbox_imgObjectScoreBeforeSplitting.isChecked())
        self.parameters["plotting_img_object_coloring_before_splitting"] = bool(self.ap_plotting_checkbox_imgObjectColoringBeforeSplitting.isChecked())
        self.parameters["plotting_img_objects_high_threshold_wave_splitting"] = bool(self.ap_plotting_checkbox_imgObjectsHighThresholdWaveSplitting.isChecked())
        self.parameters["plotting_img_object_coloring_after_splitting"] = bool(self.ap_plotting_checkbox_imgObjectColoringAfterSplitting.isChecked())
        self.parameters["plotting_img_bounding_boxes"] = bool(self.ap_plotting_checkbox_imgBoundingBoxes.isChecked())
        self.parameters["plotting_density_general"] = bool(self.ap_plotting_checkbox_imgDensityPlotGeneral.isChecked())
        self.parameters["plotting_density_preceding"] = bool(self.ap_plotting_checkbox_imgDensityPlotPreceding.isChecked())

        self.parameters["img_DPI"] = float(self.ap_plotting_lineEdit_DPI.text())

        # Update _pixel variables
        self.parameters = SM2.update_pixel_parameters(self.parameters, self.parameters["pixel_width_um"], self.parameters["lps"])

    def update_gui_from_parameters(self):
        """Update the GUI according to self.parameters."""
        parameters = self.parameters
        # Basic parameters
        self.lineEdit_parameters_basic_pixel_width.setText(str(parameters["pixel_width_um"]))
        self.lineEdit_parameters_basic_fps.setText(str(parameters["lps"]))  #
        self.lineEdit_parameters_spark_detection_threshold.setText(str(parameters["spark_detection_compound_threshold"]))
        self.lineEdit_parameters_extracellular_space_threshold.setText(str(parameters["extracellular_rejection"]))

        # Image preprocessing
        self.ap_image_preprocessing_tickboxTimeSubtraction.setChecked(bool(parameters["preprocessing_time_background_subtraction"]))
        self.ap_image_preprocessing_tickboxSpaceSubtraction.setChecked(bool(parameters["preprocessing_space_background_subtraction"]))
        self.ap_image_preprocessing_lineEdit_1dGaussFilter.setText(str(parameters["preprocessing_1dGaussFilter_um"]))
        self.ap_image_preprocessing_lineEdit_median_width.setText(str(parameters["preprocessing_medfilt_width_um"]))
        self.ap_image_preprocessing_lineEdit_median_duration.setText(str(parameters["preprocessing_medfilt_duration_ms"]))
        self.ap_image_preprocessing_lineEdit_temporal_filter_duration.setText(str(parameters["preprocessing_temporal_filter_duration"]))

        # Spark detection
        self.ap_spark_detection_lineEdit_object_threshold.setText(str(parameters["spark_detection_object_detection_threshold"]))
        self.ap_spark_detection_lineEdit_morphological_radius.setText(str(parameters["spark_detection_morpho_radius_um"]))
        self.ap_spark_detection_lineEdit_brightness_quantile.setText(str(parameters["spark_detection_quantile_level"]))
        #self.ap_spark_detection_lineEdit_minimum_pixel_density.setText(str(parameters["spark_detection_min_object_density"]))

        # Long sparks
        self.ap_long_spark_checkbox_search.setChecked(bool(parameters["long_sparks_search_for_long_sparks"]))
        self.ap_long_sparks_lineEdit_long_spark_core_diameter.setText(str(parameters["long_sparks_core_diameter_um"]))
        self.ap_long_sparks_lineEdit_minimum_duration.setText(str(parameters["long_sparks_min_long_spark_duration_ms"]))
        self.ap_long_sparks_lineEdit_long_spark_width.setText(str(parameters["long_sparks_max_long_spark_width_um"]))
        self.ap_long_sparks_lineEdit_minimum_sparkiness.setText(str(parameters["long_sparks_threshold_long_sparkiness"]))

        # Splitting sparks
        self.ap_spark_splitting_lineEdit_threshold_step.setText(str(parameters["splitting_sparks_splitting_threshold_step"]))
        self.ap_spark_splitting_lineEdit_minimum_depth.setText(str(parameters["splitting_sparks_splitting_min_split_depth"]))
        self.ap_spark_splitting_lineEdit_maximum_threshold.setText(str(parameters["splitting_sparks_splitting_max_threshold"]))

        # Wave classification
        self.ap_wave_classification_checkbox_search.setChecked(bool(parameters["wave_classification_search_for_waves"]))
        self.ap_wave_classification_lineEdit_wave_subcore_threshold.setText(str(parameters["wave_classification_wave_subcore_detection_threshold"]))
        self.ap_wave_classification_lineEdit_miniwave_score_threshold.setText(str(parameters["wave_classification_miniwave_threshold"]))
        self.ap_wave_classification_lineEdit_wave_score_threshold.setText(str(parameters["wave_classification_wave_threshold"]))

        # Scoring functions
        self.ap_scoring_lineEdit_spark_size_midpoint.setText(str(parameters["scoring_spark_scoring_size_midpoint_umms"]))
        self.ap_scoring_lineEdit_spark_size_slope.setText(str(parameters["scoring_spark_scoring_size_params_ab"][1]))
        self.ap_scoring_lineEdit_spark_brightness_midpoint.setText(str(parameters["scoring_spark_scoring_brightness_params_ab"][0]))
        self.ap_scoring_lineEdit_spark_brightness_slope.setText(str(parameters["scoring_spark_scoring_brightness_params_ab"][1]))
        self.ap_scoring_lineEdit_subspark_size_midpoint.setText(str(parameters["scoring_subspark_scoring_size_midpoint_umms"]))
        self.ap_scoring_lineEdit_subspark_size_slope.setText(str(parameters["scoring_subspark_scoring_size_params_ab"][1]))
        self.ap_scoring_lineEdit_wave_size_midpoint.setText(str(parameters["scoring_wave_scoring_size_midpoint_umms"]))
        self.ap_scoring_lineEdit_wave_size_slope.setText(str(parameters["scoring_wave_scoring_size_params_ab"][1]))
        self.ap_scoring_lineEdit_wave_brightness_midpoint.setText(str(parameters["scoring_wave_scoring_brightness_params_ab"][0]))
        self.ap_scoring_lineEdit_wave_brightness_slope.setText(str(parameters["scoring_wave_scoring_brightness_params_ab"][1]))

        # Plotting
        self.ap_plotting_checkbox_imgColNormalized.setChecked(bool(parameters["plotting_img_col_normalized"]))
        self.ap_plotting_checkbox_imgNormalized.setChecked(bool(parameters["plotting_img_normalized"]))
        self.ap_plotting_checkbox_imgNormalizedAndSmoothed.setChecked(bool(parameters["plotting_img_normalized_and_smoothed"]))
        self.ap_plotting_checkbox_imgSDtransform.setChecked(bool(parameters["plotting_img_SD_transform"]))
        self.ap_plotting_checkbox_imgCandidateObjects.setChecked(bool(parameters["plotting_img_candidate_objects"]))
        self.ap_plotting_checkbox_imgSizeRawMap.setChecked(bool(parameters["plotting_img_size_raw_map"]))
        self.ap_plotting_checkbox_imgSizeScoreMap.setChecked(bool(parameters["plotting_img_size_score_map"]))
        self.ap_plotting_checkbox_imgBrightnessQuantileMap.setChecked(bool(parameters["plotting_img_brightness_quantile_map"]))
        self.ap_plotting_checkbox_imgBrightnessScoreMap.setChecked(bool(parameters["plotting_img_brightness_score_map"]))
        self.ap_plotting_checkbox_imgObjectScoreBeforeSplitting.setChecked(bool(parameters["plotting_img_object_score_before_splitting"]))
        self.ap_plotting_checkbox_imgObjectColoringBeforeSplitting.setChecked(bool(parameters["plotting_img_object_coloring_before_splitting"]))
        self.ap_plotting_checkbox_imgObjectsHighThresholdWaveSplitting.setChecked(bool(parameters["plotting_img_objects_high_threshold_wave_splitting"]))
        self.ap_plotting_checkbox_imgDensityPlotGeneral.setChecked(bool(parameters["plotting_density_general"]))
        self.ap_plotting_checkbox_imgDensityPlotPreceding.setChecked(bool(parameters["plotting_density_preceding"]))

        self.ap_plotting_lineEdit_DPI.setText(str(self.parameters["img_DPI"]))

        self.ap_plotting_checkbox_rawImg.setChecked(bool(parameters["plotting_raw_img"]))
        self.ap_plotting_checkbox_imgObjectColoringAfterSplitting.setChecked(bool(parameters["plotting_img_object_coloring_after_splitting"]))
        self.ap_plotting_checkbox_imgBoundingBoxes.setChecked(bool(parameters["plotting_img_bounding_boxes"]))

        self.update_pixel_labels()

    def listener_sigmoid_button_clicked(self, iButton):
        """Handles clicking on one of the plotting buttons that lead to a display of the corresponding sigmoid curve.

        Parameters
        ----------
        iButton : int
            index of the button (1-based, from top to bottom)
            """
        if iButton == 1:
            midpoint = self.parameters["scoring_spark_scoring_size_params_ab"][0]
            slope = self.parameters["scoring_spark_scoring_size_params_ab"][1]
            xlabel_test = "Spark size scoring (pixels)"
        elif iButton == 2:
            midpoint = self.parameters["scoring_spark_scoring_brightness_params_ab"][0]
            slope = self.parameters["scoring_spark_scoring_brightness_params_ab"][1]
            xlabel_test = "Spark brightness scoring (SD units over threshold)"
        elif iButton == 3:
            midpoint = self.parameters["scoring_subspark_scoring_size_params_ab"][0]
            slope = self.parameters["scoring_subspark_scoring_size_params_ab"][1]
            xlabel_test = "Subspark size scoring (pixels)"
        elif iButton == 4:
            midpoint = self.parameters["scoring_wave_scoring_size_params_ab"][0]
            slope = self.parameters["scoring_wave_scoring_size_params_ab"][1]
            xlabel_test = "Wave size scoring (pixels)"
        elif iButton == 5:
            midpoint = self.parameters["scoring_wave_scoring_brightness_params_ab"][0]
            slope = self.parameters["scoring_wave_scoring_brightness_params_ab"][1]
            xlabel_test = "Wave brightness scoring (SD units over threshold)"

        ylabel_text = "score"
        function_sigmoid = lambda x: 1 - 1/(1 + (x/midpoint)**slope)

        start_x = 0
        end_x = midpoint * 99**(1/slope) # this is obtained by solving the equation of the sigmoid being equal to 0.99

        x = numpy.linspace(start_x, end_x, 1000)

        fig = plt.figure()
        ax = plt.axes()
        ax.plot(x, function_sigmoid(x))
        plt.xlabel(xlabel_test)
        plt.ylabel(ylabel_text)
        plt.show()

    def listener_browse_button_img_saving_folder_clicked(self):
        """Handles the button giving the user the option to choose the folder where images created during spark analysis are stored."""
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Folder')
        if len(folder_path) > 0:
            self.parameters["img_output_folder"] = str(folder_path)
            self.ap_plotting_lineEdit_folder_save.setText(str(folder_path))

    def listener_lineEdit_folder_save_edited(self):
        """Handles manual editing of the text window with the path to the folder where images created during spark analysis are stored."""
        self.parameters["img_output_folder"] = str(self.ap_plotting_lineEdit_folder_save.text())
        self.ap_plotting_lineEdit_folder_save.setText(str(self.parameters["img_output_folder"]))

    def listener_button_load_image(self):
        """Handles the user selecting an image to be analyzed."""
        fname = []
        fname = QFileDialog.getOpenFileName(self,  "Select an image file")

        self.button_load_image.setDown(True)
        if exists(fname[0]):
            print(["Reading "+fname[0]])
            img = plt.imread(fname[0])

            # Handling the case when an RGB image is provided
            if len(img.shape) > 2:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setText("The image read is not single-channel, but contains multiple color channel data.")
                msg.setInformativeText("SM2 is trying to guess which color channel to use, and you will see which one was chosen after hitting OK. If this is a different channel from the one you want analyzed, please make a single-channel copy of your image, containing only the color channel with sparks.")
                msg.setWindowTitle("RGB image detected")
                msg.exec_()
                sdPerChannel = np.zeros((img.shape[2], 1))
                for iChannel in range(len(img.shape)):
                    sdPerChannel[iChannel] = np.nanstd(img[:,:,iChannel])

                img = img[:,:, np.argmax(sdPerChannel)]  # We choose the channel with the highest standard deviation.

            self.current_image = img
            self.current_image_fname = os.path.basename(fname[0])
            plt.figure()  # After reading an image, it is shown
            plt.imshow(img, cmap="gray")
            plt.title("Image read")
            plt.grid(False)
            plt.show()

        self.button_load_image.setDown(False)
        #plt.figure()
        #plt.imshow(current_image)
        #plt.show()
        #print(fname)

    def listener_button_analyze_data(self):
        "Handles the user requesting spark analysis and feature extraction to be performed."
        self.button_analyze_data.setDown(True)
        if len(self.current_image) > 0:
            try:
                numbered_mask_after_split, img_den_gauss, img_den_not_normalized, img_SD_multiple, waves, miniwaves, long_sparks_numbering = SM2.segment_sparks(self.current_image, self.parameters)
            except Exception as inst:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setText(str(inst))
                msg.setWindowTitle("Error")
                msg.exec_()
                return

            self.analysis_outputs, self.analysis_outputs_summary, __ = SM2.analyze_sparks(numbered_mask_after_split, img_den_not_normalized,  waves, miniwaves, long_sparks_numbering, self.parameters)
            self.analysis_outputs_summary['filename'] = self.current_image_fname
            plt.show()
            print("analysis done")
        else:
            print("An image needs to be read first.")
        self.button_analyze_data.setDown(False)

    def listener_button_save_outputs(self):
        """Handles the user requesting to store the analysis outputs.

        This function stores two files: one with each row corresponding to a single spark, and a summary file, where median, Q25, and Q75 are given for the whole recording."""
        self.button_save_outputs.setDown(True)
        file_name, param2 = QFileDialog.getSaveFileName(self, "Save analysis outputs", "", "CSV file (*.csv);;MS Excel (*.xlsx)")

        if len(file_name) > 0:
            split_tup = os.path.splitext(file_name)

            fname = split_tup[0]
            file_extension = split_tup[1]

            if hasattr(self.analysis_outputs, 'to_csv'):
                if file_extension == '.xlsx':
                    self.analysis_outputs.to_excel(file_name)
                    self.analysis_outputs_summary.to_excel(file_name.replace(".", "_summary."))
                else:
                    self.analysis_outputs.to_csv(file_name)
                    self.analysis_outputs_summary.to_csv(file_name.replace(".", "_summary."))
                print("saving")
        self.button_save_outputs.setDown(False)


    def listener_button_autofit(self):
        """Handles the autofit button click, either running simple parameter search (varying the spark detection threshold), or a more complex one, which can vary any parameter,
        using a csv file to define which parameter combinations are to be explored."""

        # Handling when the autofit button is hit

        # Request files to load.
        file_name = QFileDialog()
        file_name.setFileMode(QFileDialog.ExistingFiles)
        to_process = file_name.getOpenFileNames(self, "Select image files")
        autofit_image_fnames = []

        # Reading the images - both to be annotated, and reference annotation
        imgs = []
        imgs_reference = []
        for fname in to_process[0]:
            filename, file_extension = os.path.splitext(fname)
            fn_reference = filename + "_segmented" + file_extension
            try:
                img = plt.imread(fname)
                img_reference = plt.imread(fn_reference)
                imgs.append(img)
                imgs_reference.append(img_reference)
            except FileNotFoundError as inst:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setText("Problem reading image or the reference file")
                msg.setInformativeText("Make sure the filename is correct and there is a similarly named file containing '_segmented' before the extension")
                msg.setWindowTitle("Error")
                msg.setDetailedText(str(inst))
                msg.exec_()
                return

        is_simple = self.radioButton.isChecked()
        params_to_explore = []
        if is_simple:
            # Generate a small window that asks for parameters
            self.autofit_basic_imgs = imgs
            self.autofit_basic_imgs_reference = imgs_reference
            self.dialog.show()
            # Hitting the OK button in the dialog takes care of the rest
        else: # more complex autofit, where image files are selected, as well as a file containing a table of parameters. This has a predefined header that is used to facilitate filling of a parameter structure
            file_parameters = QFileDialog()
            file_parameters.setFileMode(QFileDialog.ExistingFiles)
            fname_parameters = file_parameters.getOpenFileNames(self, "Select a csv file defining the parameters")
            if len(fname_parameters[0]) == 0: # Just handling the case when no file is selected
                return

            fname_parameters = fname_parameters[0][0]

            # Generate a structure of parameters
            param_matrix = pd.read_csv(fname_parameters)
            n_parameter_combinations = param_matrix.shape[0]
            penalty_fp = np.zeros((n_parameter_combinations, 1))
            penalty_fn = np.zeros((n_parameter_combinations, 1))
            min_dice = np.zeros((n_parameter_combinations, 1))

            for i_param in range(n_parameter_combinations):
                params_to_explore.append(copy.deepcopy(self.parameters))

            for i_param in range(n_parameter_combinations):
                # For each parameter, we work on its copy, disabling plotting, and updating any values according to the parameter-defining file

                # Go over all the parameters in the file - if any is non-nan, replace its corresponding entry in param
                # also we're using SPEC_penalty_fp and _fn to define these
                params_to_explore[i_param]["plotting_raw_img"] = 0
                params_to_explore[i_param]["plotting_img_col_normalized"] = 0
                params_to_explore[i_param]["plotting_img_normalized"] = 0
                params_to_explore[i_param]["plotting_img_normalized_and_smoothed"] = 0
                params_to_explore[i_param]["plotting_img_SD_transform"] = 0
                params_to_explore[i_param]["plotting_img_candidate_objects"] = 0
                params_to_explore[i_param]["plotting_img_size_raw_map"] = 0
                params_to_explore[i_param]["plotting_img_size_score_map"] = 0
                params_to_explore[i_param]["plotting_img_brightness_quantile_map"] = 0
                params_to_explore[i_param]["plotting_img_brightness_score_map"] = 0
                params_to_explore[i_param]["plotting_img_object_score_before_splitting"] = 0
                params_to_explore[i_param]["plotting_img_object_coloring_before_splitting"] = 0
                params_to_explore[i_param]["plotting_img_objects_high_threshold_wave_splitting"] = 0
                params_to_explore[i_param]["plotting_img_object_coloring_after_splitting"] = 0
                params_to_explore[i_param]["plotting_img_bounding_boxes"] = 0

                penalty_fp[i_param] = 1  # default value
                penalty_fn[i_param] = 1
                min_dice[i_param] = 0.15
                for key in param_matrix.keys():
                    if not(np.isnan(param_matrix[key][i_param])): # We continue only if the parameter value is filled
                        if len(key) >= 5:  # This is just for safety, it should always hold
                            if key[0:5] == 'SPEC_':
                                if key == 'SPEC_penalty_fp':
                                    penalty_fp[i_param] = param_matrix[key][i_param]
                                elif key == 'SPEC_penalty_fn':
                                    penalty_fn[i_param] = param_matrix[key][i_param]
                                elif key == 'SPEC_min_dice':
                                    min_dice[i_param] = param_matrix[key][i_param]
                                elif key == 'SPEC_spark_size_slope':
                                    params_to_explore[i_param]["scoring_spark_scoring_size_params_ab"][1] = param_matrix[key][i_param]
                                elif key == 'SPEC_subspark_size_slope':
                                    params_to_explore[i_param]["scoring_subspark_scoring_size_params_ab"][1] = param_matrix[key][i_param]
                                elif key == 'SPEC_spark_brightness_midpoint':
                                    params_to_explore[i_param]["scoring_spark_scoring_brightness_params_ab"][0] = param_matrix[key][i_param]
                                elif key == 'SPEC_spark_brightness_slope':
                                    params_to_explore[i_param]["scoring_spark_scoring_brightness_params_ab"][1] = param_matrix[key][i_param]
                                elif key == 'SPEC_wave_scoring_size_slope':
                                    params_to_explore[i_param]["scoring_wave_scoring_size_params_ab"][1] = param_matrix[key][i_param]
                                elif key == 'SPEC_wave_brightness_midpoint':
                                    params_to_explore[i_param]["scoring_wave_scoring_brightness_params_ab"][0] = param_matrix[key][i_param]
                                elif key == 'SPEC_wave_brightness_slope':
                                    params_to_explore[i_param]["scoring_wave_scoring_brightness_params_ab"][1] = param_matrix[key][i_param]
                            else:  # now we're handling a common parameter
                                params_to_explore[i_param][key] = param_matrix[key][i_param]

                # Updating pixel-based values
                params_to_explore[i_param] = SM2.update_pixel_parameters(params_to_explore[i_param], params_to_explore[i_param]["pixel_width_um"], params_to_explore[i_param]["lps"])
                #print('ok')

            # Running the evaluation of the parameter combinations on the images
            n_fp, n_fn, error_scores = SM2.evaluate_list_of_parameters(imgs, imgs_reference, params_to_explore, penalty_fp, penalty_fn, min_dice)

            # Summing the errors, fp, and fn over multiple files
            if len(imgs) > 1:
                n_fp = np.sum(n_fp, axis=1)
                n_fn = np.sum(n_fn, axis=1)
                error_scores = np.sum(error_scores, axis=1)
            # Plotting the performance curves
            param_indices = list(range(1, len(params_to_explore)+1))
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
            fig.tight_layout()
            fig.suptitle("Performance versus threshold (sum over input files)")
            ax1.plot(param_indices, error_scores)
            ax1.set_title("# overall error score")
            ax2.plot(param_indices, n_fp)
            ax2.set_title("# false positives")
            ax3.plot(param_indices, n_fn)
            ax3.set_title("# false negatives")
            plt.show()
            #print('ok')

    def listener_button_batch_analysis(self):
        """Handles the batch analysis, running spark analysis & feature extraction for multiple files.

        It stores in the user-selected folder a single file for each recording (one line ~ one spark), and a single summary file (one row ~ one file summary)
        """
        file_name = QFileDialog()
        file_name.setFileMode(QFileDialog.ExistingFiles)
        batch_fnames = file_name.getOpenFileNames(self, "Select image files")
        batch_fnames = batch_fnames[0]

        out_path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select output folder')
        if not(exists(out_path)):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("Make sure that the target folder exists")
            msg.setWindowTitle("Error")
            msg.exec_()
            return

        batch_analysis_summaries = [] # each entry will be a dataframe corresponding to a summary
        n_files = len(batch_fnames)
        i_fname = 0
        for fname in batch_fnames:
            print('Processing file no. '+str(i_fname+1)+'/'+str(n_files))

            img = plt.imread(fname)

            # getting
            split_tup = os.path.splitext(fname)
            extension = split_tup[1]
            basic_fname = os.path.basename(fname)
            fname_stem = Path(fname).stem # only the filename, no extension, no path

            params = copy.deepcopy(self.parameters) # if we're storing images, we make a subfolder there
            if exists(params["img_output_folder"]):
                params["img_output_folder"] = os.path.join(params["img_output_folder"], fname_stem)
                Path(os.path.join(params["img_output_folder"])).mkdir(exist_ok=True)

            try:
                numbered_mask_after_split, img_den_gauss, img_den_not_normalized, img_SD_multiple, waves, miniwaves, long_sparks_numbering = SM2.segment_sparks(img, params)
            except Exception as inst:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setText(str(inst))
                msg.setWindowTitle("Error")
                msg.exec_()
                return

            analysis_outputs, analysis_outputs_summary,__ = SM2.analyze_sparks(numbered_mask_after_split, img_den_not_normalized, waves, miniwaves, long_sparks_numbering, params)

            # saving the csv with data on sparks

            analysis_outputs.to_csv(os.path.join(out_path, basic_fname.replace(extension, '.csv')))

            # saving the summary
            analysis_outputs_summary['filename'] = basic_fname
            batch_analysis_summaries.append(analysis_outputs_summary)
            #analysis_outputs_summary.to_csv(file_name.replace(".", "_summary."))
            i_fname = i_fname + 1
            plt.show()

        summary_all = pd.concat(batch_analysis_summaries)
        summary_all.to_csv(os.path.join(out_path, 'batch_summary.csv'))
        print('batch finished')

    def listener_button_save_parameters(self):
        """Saves current parameters as a pickle file."""
        dialog = QFileDialog()
        dialog.setDefaultSuffix('pkl')
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        dialog.setNameFilters(['Parameters (*.pkl)'])
        dialog.exec()

        file_name = dialog.selectedFiles() #QFileDialog().getSaveFileName(self, 'Select the filename for saving the file')
        if len(file_name)>0:
            f = open(file_name[0], "wb")
            copyParams = copy.deepcopy(self.parameters)
            # For saving, we remove the lambda functions
            try:
                copyParams.pop("scoring_function_spark_scoring_size_pixels")
                copyParams.pop("scoring_function_subspark_scoring_size_pixels")
                copyParams.pop("scoring_function_spark_scoring_brightness")
                copyParams.pop("scoring_function_wave_scoring_size_pixels")
                copyParams.pop("scoring_function_wave_scoring_brightness")
            except Exception as inst:
                print('')

            pickle.dump(copyParams, f)
            f.close()
            print("parameters saved")

    def listener_button_load_parameters(self):
        """Loads parameters from a pickle file."""
        dialog = QFileDialog()
        dialog.setDefaultSuffix('pkl')
        dialog.setAcceptMode(QFileDialog.AcceptOpen)
        dialog.setNameFilters(['Parameters (*.pkl)'])
        dialog.exec()

        file_name = dialog.selectedFiles()
        if len(file_name) > 0:
            if exists(file_name[0]):
                with (open(file_name[0], "rb")) as openfile:
                    pickled_params = pickle.load(openfile)
                    self.is_initializing = 1 # This we need to do to prevent a strange "conflict" of listeners...
                    self.parameters = pickled_params
                    self.update_gui_from_parameters()
                    self.is_initializing = 0
                    print("parameters loaded")

                # # And we fill lambda functions again
                # self.parameters["scoring_function_spark_scoring_size_pixels"] = lambda x: 1 - 1. / (
                #             1 + (x / self.parameters["scoring_spark_scoring_size_params_ab"][0]) ** self.parameters["scoring_spark_scoring_size_params_ab"][1])
                #
                # self.parameters["scoring_function_subspark_scoring_size_pixels"] = lambda x: 1 - 1. / (
                #             1 + (x / self.parameters["scoring_subspark_scoring_size_params_ab"][0]) ** self.parameters["scoring_subspark_scoring_size_params_ab"][1])
                #
                #
                # self.parameters["scoring_function_spark_scoring_brightness"] = lambda x: 1 - 1. / (
                #             1 + (x / self.parameters["scoring_spark_scoring_brightness_params_ab"][0]) ** self.parameters["scoring_spark_scoring_brightness_params_ab"][1])
                #
                #
                # self.parameters["scoring_function_wave_scoring_size_pixels"] = lambda x: 1 - 1. / (
                #             1 + (x / self.parameters["scoring_wave_scoring_size_params_ab"][0]) ** self.parameters["scoring_wave_scoring_size_params_ab"][1])
                #
                #
                # self.parameters["scoring_function_wave_scoring_brightness"] = lambda x: 1 - 1. / (
                #             1 + (x / self.parameters["scoring_wave_scoring_brightness_params_ab"][0]) ** self.parameters["scoring_wave_scoring_brightness_params_ab"][1])

                # fill GUI elements based on the parameters


    def listener_param_fitting_basic_dialog_clicked(self):
        """When basic parameter fitting is performed, a popup lets the user select additional parameters - once that is done, hitting the OK button is handled by this function, which does the simple fitting."""
        imgs = self.autofit_basic_imgs
        imgs_reference = self.autofit_basic_imgs_reference
        # parse parameters from the dialog.
        threshold_from = float(self.dialog.le_threshold_from.text())
        threshold_step = float(self.dialog.le_threshold_step.text())

        threshold_to = float(self.dialog.le_threshold_to.text()) + threshold_step/2  # We increase this variable by threshold_step/2 to make sure that the threshold_to value in GUI is processed

        minimum_dice_overlap = float(self.dialog.le_min_dice.text())
        false_positive_penalty = float(self.dialog.le_false_positive_penalty.text())
        false_negative_penalty = float(self.dialog.le_false_negative_penalty.text())
        n_parameter_combinations = len(np.arange(threshold_from, threshold_to+0.00000000001, threshold_step))
        penalty_fp = false_positive_penalty * np.ones((n_parameter_combinations, 1))
        penalty_fn = false_negative_penalty * np.ones((n_parameter_combinations, 1))
        min_dice = minimum_dice_overlap * np.ones((n_parameter_combinations, 1))

        params_to_explore = []
        for i_param in range(n_parameter_combinations):
            params_to_explore.append(copy.deepcopy(self.parameters))

        for i_param in range(n_parameter_combinations):
            # For each parameter, we work on its copy, disabling plotting, and updating the compound threshold
            params_to_explore[i_param]["plotting_raw_img"] = 0
            params_to_explore[i_param]["plotting_img_col_normalized"] = 0
            params_to_explore[i_param]["plotting_img_normalized"] = 0
            params_to_explore[i_param]["plotting_img_normalized_and_smoothed"] = 0
            params_to_explore[i_param]["plotting_img_SD_transform"] = 0
            params_to_explore[i_param]["plotting_img_candidate_objects"] = 0
            params_to_explore[i_param]["plotting_img_size_raw_map"] = 0
            params_to_explore[i_param]["plotting_img_size_score_map"] = 0
            params_to_explore[i_param]["plotting_img_brightness_quantile_map"] = 0
            params_to_explore[i_param]["plotting_img_brightness_score_map"] = 0
            params_to_explore[i_param]["plotting_img_object_score_before_splitting"] = 0
            params_to_explore[i_param]["plotting_img_object_coloring_before_splitting"] = 0
            params_to_explore[i_param]["plotting_img_objects_high_threshold_wave_splitting"] = 0
            params_to_explore[i_param]["plotting_img_object_coloring_after_splitting"] = 0
            params_to_explore[i_param]["plotting_img_bounding_boxes"] = 0

            params_to_explore[i_param]["spark_detection_compound_threshold"] = threshold_from + i_param * threshold_step

        # Running the evaluation of the parameter combinations on the images
        n_fp, n_fn, error_scores = SM2.evaluate_list_of_parameters(imgs, imgs_reference, params_to_explore, penalty_fp, penalty_fn, min_dice)

        # Summing the errors, fp, and fn over multiple files
        if len(imgs) > 1:
            n_fp = np.sum(n_fp, axis=1)
            n_fn = np.sum(n_fn, axis=1)
            error_scores = np.sum(error_scores, axis=1)
        # Plotting the performance curves
        param_indices = list(range(1, len(params_to_explore) + 1))
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        fig.tight_layout()
        fig.suptitle("Performance versus threshold (sum over input files)")
        ax1.plot(param_indices, error_scores)
        ax1.set_title("# overall error score")
        ax2.plot(param_indices, n_fp)
        ax2.set_title("# false positives")
        ax3.plot(param_indices, n_fn)
        ax3.set_title("# false negatives")
        plt.show()
        self.dialog.hide()

    def listener_button_close_all_figures(self):
        """Closes all open figures."""
        plt.close('all')
        gc.collect()
        print("closed all open figures")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.show()
    sys.exit(app.exec_())

