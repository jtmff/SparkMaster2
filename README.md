# SparkMaster2
SparkMaster 2 (SM2) software for the analysis of Ca sparks. This software is licensed under the GNU GPL v3. 
If this is a problem for your application, please get in touch with us at jakub.tomek.mff@gmail.com

*** Files
/SM2_source contains the source codes for SM2
gui.py - the python code for the main GUI of SM2 (but does not run it, main.py serves that purpose).
gui.ui - a file defining the GUI structure. Can be opened via Qt designer.
main.py - the file to make the SM2 gui show and run.
sampleScriptProcessing.py - an example of how you can use SM2 from Python with a script.
simple_param_popup.py - a small window used for simple parameter autofitting.
simple_param_popup.ui - a file defining the structure of the above.
SM2.py - the core functionality of SM2. This can be accessed either via a GUI, or directly (as shown in sampleScriptProcessing).

/synthetic_spark_generator contains (Matlab) source codes for the generation of synthetic spark images.
makeSyntheticDataGray.m - creates a set of images
sparkLibrary.m - a set of representations of several real sparks found in our data.

*** Compilation
To compile SM2 into an executable file, we recommend using pyinstaller (after you install all the required libraries, the ones imported in main.py and SM2.py).

The command to use is:
pyinstaller -F --hidden-import=imagesc --hidden-import=scikit-image --hidden-import=sklearn.utils._typedefs --hidden-import=sklearn.neighbors._partition_nodes --onefile --noconsole main.py

For Mac, it is however necessary to further uncomment the two following lines in main.py before compiling.
import matplotlib
matplotlib.use(‘qtagg’)