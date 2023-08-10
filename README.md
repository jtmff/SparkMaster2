# SparkMaster 2
SparkMaster 2 (SM2) software for the analysis of Ca sparks. 
This software is licensed under the GNU GPL v3. If you have any questions (scientific, as well as on licensing) please get in touch with us at jakub.tomek.mff@gmail.com

## Runnable application download
The runnable version of SM2 for Windows, macOS, and Linux is available at the [UC Davis site](https://somapp.ucdmc.ucdavis.edu/Pharmacology/bers/) or alternatively [Google Drive](https://drive.google.com/drive/folders/1Gs_f9ilt5Orq9AeqWzHas44Pz8-UKZaV?usp=sharing), including sample data and a user guide. Please see the **README** file there if you encounter any issues downloading/running SM2. In particular, if you want to use the Mac version, it will point you to an explanation of how to make the application bypass the overzealous macOS security (it is simple do not worry).

## Files
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

## Compilation
If you want to compile the SM2 source codes into a runnable file, we recommend using pyinstaller (after you install all the required libraries: the ones imported in main.py and SM2.py).

The command to use is:
pyinstaller -F --hidden-import=imagesc --hidden-import=scikit-image --hidden-import=sklearn.utils._typedefs --hidden-import=sklearn.neighbors._partition_nodes --onefile --noconsole main.py

For Mac, it is however necessary to further uncomment the two following lines in main.py before compiling.
import matplotlib
matplotlib.use(‘qtagg’)
