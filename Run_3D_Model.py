'''
3D Greenland Ice Sheet Meltwater Model Shell - Lydia Gilbert
'''
import sys
from datetime import date
import pandas as pd
import meltwater3dmodel
from meltwater3dmodel import Model
import matplotlib.pyplot as plt
import f90nml
import cProfile
import pstats
import os
import numpy as np

'''
This script runs the 3D model built in the meltwater3dmodel.py script. It takes command line arguments:
help - prints list of command line arguments & info about model, then quits
[input file name].nml - fortran namelist file of input arguments

note: cannot have gaps in data between the startdate and enddate (both included)
'''

# Prints out help text
if (sys.argv[1] == 'help'):
    print('This script runs the 3D model built in the meltwater3dmodel.py script.')
    print('It takes command line arguments:')
    print('help - prints list of command line arguments & info about model, then quits')
    print('[input file name].nml - fortran namelist file of input arguments')
    sys.exit()

# Reads in namelist file
inputs = f90nml.read(sys.argv[1])

# Extracts variables for threshold, moulins and graphing from namelist variable
threshold = inputs['input_nml']['threshold']
moulin = inputs['input_nml']['moulin']
prof = inputs['input_nml']['profile']
G3 = inputs['input_nml']['G3']
G2 = inputs['input_nml']['G2']
G2Z = inputs['input_nml']['G2Z']
GS = inputs['input_nml']['GS']
GSS = inputs['input_nml']['GSS']
filepath_to_graphs = inputs['input_nml']['filepath_to_graphs']

# If run the profiler is set to true
if (prof):
    # Initialize & start cProfile
    profiler = cProfile.Profile()
    profiler.enable()

# If filepath to graphs doesn't exist, make directory
if not (os.path.exists(filepath_to_graphs)):
    os.makedirs(filepath_to_graphs)

# Sets up and runs model based on provided namelist parameters
model_run = Model(inputs['input_nml']['filepath_to_DEM'], inputs['input_nml']['filepath_to_surf_melt'], 
                  inputs['input_nml']['filepath_to_output'], threshold, moulin)

model_run.run(inputs['input_nml']['startdate'], inputs['input_nml']['enddate'])

# If namelist file indicates 3D graphs are to be made
if (G3[0]):
    # If plotting 3D graphs
    print('Making 3D graphs...')
    start_date = date(G3[3], G3[1], G3[2])
    end_date = date(G3[6], G3[4], G3[5])
    date_range = pd.date_range(start_date, end_date)
    for i in range(0, len(date_range)):
        model_run.plot_model3D(date_range[i])
        plt.savefig(filepath_to_graphs+'Graph3D_'+str(threshold)+'t_'+str(date_range[i].month)+'_'+str(date_range[i].day)+'_2019.eps')
        plt.close()
    
    
# If namelist file indicates 2D graphs are to be made
if (G2[0]):
    # If plotting 2D graphs
    print('Making 2D graphs...')
    start_date = date(G2[3], G2[1], G2[2])
    end_date = date(G2[6], G2[4], G2[5])
    date_range = pd.date_range(start_date, end_date)
    #date_range = pd.Series(index=np.arange(0,4), data=pd.to_datetime([date(2019,  5, 9), date(2019, 6, 10), date(2019, 7, 12), date(2019, 8, 13)]))
    for i in range(0, len(date_range)):
        model_run.plot_model2D(date_range[i])
        #plt.savefig(filepath_to_graphs+'Graph2D_'+str(threshold)+'t_'+str(date_range[i].month)+'_'+str(date_range[i].day)+'_2019.eps')
        # Lines to make graphs for animations
        n = str(i+1)
        if (moulin > 0):
            plt.savefig(filepath_to_graphs+'Graph2D_'+str(threshold)+'t_'+str(moulin)+'m_'+n.zfill(4)+'.jpg',dpi=400)
        else:
            plt.savefig(filepath_to_graphs+'Graph2D_'+str(threshold)+'t_'+n.zfill(4)+'.jpg',dpi=400)
        plt.close()

if (G2Z[0]):
    # If plotting 2D graphs
    print('Making Zoomed 2D graphs...')
    start_date = date(G2Z[7], G2Z[5], G2Z[6])
    end_date = date(G2Z[10], G2Z[8], G2Z[9])
    date_range = pd.date_range(start_date, end_date)
    for i in range(0, len(date_range)):
        model_run.plot_zoom_model2D(date_range[i], G2Z[1], G2Z[2], G2Z[3], G2Z[4])
        #plt.savefig(filepath_to_graphs+'Graph2D_'+str(threshold)+'t_'+str(date_range[i].month)+'_'+str(date_range[i].day)+'_2019.eps')
        # Lines to make graphs for animations
        n = str(i+1)
        if (moulin > 0):
            plt.savefig(filepath_to_graphs+'Graph2ZD_'+str(threshold)+'t_'+str(moulin)+'m_'+'r['+str(G2Z[1])+'-'+str(G2Z[2])+']_c['+str(G2Z[3])+'-'+str(G2Z[4])+']_'+n.zfill(4)+'.jpg',dpi=400)
        else:
            plt.savefig(filepath_to_graphs+'Graph2ZD_'+str(threshold)+'t_'+'r['+str(G2Z[1])+'-'+str(G2Z[2])+']_c['+str(G2Z[3])+'-'+str(G2Z[4])+']_'+n.zfill(4)+'.jpg',dpi=400)
        plt.close()

# If namelist file indicates slice graphs are to be made
if (GS[0]):
    # If plotting slice graphs
    print('Making slice graphs...')
    start_date = date(GS[4], GS[2], GS[3])
    end_date = date(GS[7], GS[5], GS[6])
    date_range = pd.date_range(start_date, end_date)
    for i in range(0, len(date_range)):
        n = str(i+1)
        model_run.plot_slice(date_range[i], GS[1])
        #plt.savefig(filepath_to_graphs+'GraphS_'+str(threshold)+'t_'+str(GS[1])+'r_'+str(date_range[i].month)+'_'+str(date_range[i].day)+'_2019.eps')
        plt.savefig(filepath_to_graphs+'GraphS_'+str(threshold)+'t_'+str(GS[1])+'r_'+n.zfill(4)+'.jpg',dpi=400)
        plt.close()

# If namelist file indicates subslice graphs are to be made
if (GSS[0]):
    # If plotting subsection of slice graphs
    print('Making subslice graphs...')
    start_date = date(GSS[6], GSS[4], GSS[5])
    end_date = date(GSS[9], GSS[7], GSS[8])
    date_range = pd.date_range(start_date, end_date)
    for i in range(0, len(date_range)):
        model_run.plot_sub_slice(date_range[i], GSS[1], GSS[2], GSS[3])
        # plt.savefig(filepath_to_graphs+'GraphSS_'+str(threshold)+'t_'+str(GS[1])+'r['+str(GSS[2])+'s'+str(GSS[3])+']_'+str(date_range[i].month)+'_'+str(date_range[i].day)+'_2019.eps')
        # Lines to make graphs for animations
        n = str(i+1)
        plt.savefig(filepath_to_graphs+'GraphSS_'+str(threshold)+'t_'+str(GS[1])+'r['+str(GSS[2])+'s'+str(GSS[3])+']_'+n.zfill(4)+'.jpg', dpi=700)
        plt.close()

# If run the profiler is set to true
if (prof):
    # Disable profiler & print results
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    print('Compiling script statistics...')
    stats.print_stats(30)