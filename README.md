# MeltwaterModel
This code runs the meltwater model described in Gilbert & Bassis. The Python packages required to run the model are listed below. Descriptions of the files are included below and in the files themselves.

## Python Packages
    numpy: 1.20.1
    osgeo: 3.3.2
    scipy: 1.7.1
    matplotlib: 3.3.4
    pyproj: 3.1.0
    pandas: 1.2.3
    cartopy: 0.20.0
    f90nml: 1.2

## Files
meltwater3dmodel.py:
    The dataclass that runs the meltwater model. Includes functions for running the model and graphing the results.

Run_3D_Model.py:
    The script that runs the meltwater3dmodel. Takes inputs from a namelist file.

Model3DInput.nml:
    Namelist file that includes input parameters to the meltwater model. Includes descriptions of the parameters and filepaths.

Surface_melt_Abl_Files:
    Input meltwater files for the meltwater model.

arctic_dem_trim.tif:
    Input elevation model file for the meltwater model.