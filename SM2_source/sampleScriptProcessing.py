import SM2
import matplotlib.pyplot as plt
import numpy as np

# Path to an image to be analyzed
filename = 'testImages/mousePermeabilized1.jpg' #"testImages/mousePermeabilized1.jpg"

# Defining recording spatiotemporal resolution
pixel_width_um = 0.12  # um
lines_per_second = 500  # ms

# This function call retrieves a dictionary of parameters (mapping parameter names to values)
params = SM2.get_default_parameters(pixel_width_um, lines_per_second)  # Getting default parameters of SM2
params["plotting_raw_img"] = 1  # Requesting that the raw image is to be displayed. See SM2.get_default_parameters for a list of parameters.

# Reading the image to be analyzed
img = plt.imread(filename)

# Detecting sparks and other Ca release objects.
# spark_mask is the mask where spark_mask[i,j] is the index of a detected object (or background).
# img_denoised is the source image after pre-processing
# waves, miniwaves, and long_sparks are lists of indices of detected objects that are judged to belong to those classes of objects
print("Printing time spent in various procedures: ")
spark_mask, _, img_denoised, _, waves, miniwaves, long_sparks = SM2.segment_sparks(img, params)

# Properties of Ca release events are recorded for each event (such as duration, width, time to peak, etc.)
# analysis outputs - a dataframe describing each event's properties
# analysis_outputs_summary - a single-line summary of the preceding dataframe, containing medians and 25- and 75- percentile values
# event_traces - traces of the release events that were analyzed
analysis_outputs, analysis_outputs_summary, event_traces = SM2.analyze_sparks(spark_mask, img_denoised, waves, miniwaves, long_sparks, params)

# Printing the median tau of decay. Print analysis_outputs.keys() for a list of feature names that can be retrieved like this.
print("Printing the median tau of decay in ms: "+str(np.median((analysis_outputs["tau of decay (ms)"]))))

# Saving the outputs as a csv
analysis_outputs.to_csv('spark_properties.csv')  # Saving a table of spark properties (e.g., amplitude, duration, etc.)

# Show the figures with visualizations
plt.show()
