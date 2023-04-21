'''
3D Greenland Ice Sheet Meltwater Model Module - Lydia Gilbert
'''
import sys

from numpy.core.defchararray import zfill
from osgeo import gdal
from osgeo import osr
import numpy as np
import scipy.integrate as integrate
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import glob
import os.path
from os import path
import os
from pyproj import Transformer, transform
from pyproj.transformer import AreaOfInterest
from dataclasses import dataclass
from datetime import datetime
from datetime import date
import pandas as pd
import cartopy.crs as ccrs

'''
This script contains the dataclass that runs the 3D meltwater model and graphs the results.
'''

@dataclass
class Point:
    '''
    This dataclass records the direction water is supposed to flow for a given point on a given iteration and 
    whether the point is on an edge or not 

    Key to direction:
    32 64 128
    16     1   0 == pool
    8   4  2   None == flow off edge
    '''
    direction: int
    edge: bool

class Model:
    '''
    Parameters to model:
    filepath_to_DEM: filepath to DEM to be used for model
    filepath_to_surfc: filepath to folder containing surface meltwater files, in increments of a single day
    filepath_to_files: filepath to folder where raster of files of model water levels output are to be stored
    threshold: value in meters of threshold of maximum absolute difference allowed between two iterations of same day
    moulin: value in meters as the threshold of water depth for moulin formation, if negative, means moulins not enabled in model
    '''
    def __init__(self, filepath_to_DEM, filepath_to_surfc, filepath_to_files, threshold, moulin):
        '''
        Initializes instance of model class given parameters
        '''
        # Print status message
        print('Initializing 3D meltwater model instance...')

        # Filepath to folder that stores all final water values, one file for each day
        # If it doesn't exist, make directory
        if not (os.path.exists(filepath_to_files)):
            os.makedirs(filepath_to_files)
        self.path_to_files = filepath_to_files

        # DEM variable that holds elevation model array that the model runs on
        src = gdal.Open(filepath_to_DEM)
        self.DEM = np.array(src.GetRasterBand(1).ReadAsArray())
        self.DEM = np.where(self.DEM < 2000, self.DEM, 1000) # Remove anomolous data

        # Create 1D list of indexes for DEM/water arrays
        indexes_1D = np.argsort(self.DEM, axis=None)
        indexes_1D = np.flip(indexes_1D)
        self.indexes = np.unravel_index(indexes_1D, self.DEM.shape)

        # Set up change of projection for graphing
        Transform_web_to_GCS = Transformer.from_crs("EPSG:3857", "EPSG:4326", area_of_interest=AreaOfInterest(-45, 69, 65, -50))

        # Create 1D & 2D arrays of latitude & longitude values and save them
        width = src.RasterXSize
        height = src.RasterYSize
        gt = src.GetGeoTransform()
        minx = gt[0]
        miny = gt[3] + width*gt[4] + height*gt[5] 
        maxx = gt[0] + width*gt[1] + height*gt[2]
        maxy = gt[3]
        self.long_1D = np.linspace(minx, maxx, width)
        self.lat_1D = np.linspace(miny, maxy, height)
        self.long_2D_utm, self.lat_2D_utm = np.meshgrid(self.long_1D, self.lat_1D)
        self.lat_2D_deg, self.long_2D_deg = Transform_web_to_GCS.transform(self.long_2D_utm, self.lat_2D_utm)
        np.save(filepath_to_files+'Long_2D_mesh.npy', self.long_2D_deg)
        np.save(filepath_to_files+'Lat_2D_mesh.npy', self.lat_2D_deg)

        # Threshold value for maximum allowable difference between two water rasters for a single day (meters)
        self.threshold = threshold

        # Water depth at which moulin formation is triggered (meters), if value is negative, no moulins in model
        if (moulin < 0):
            self.moulin_use = False

            # Print status message
            print('Disabling moulins...')
        else:
            self.moulin_use = True
            self.moulin_thresh = moulin
            self.moulin = np.full(self.DEM.shape,False)

            # Print status message
            print('Enabling moulins...')

        # Water holds all the water values for a single iteration of the model, same size as DEM
        self.water = np.zeros(self.DEM.shape)
        self.updated_ras = np.zeros(self.DEM.shape)

        # Data is same shape as DEM and holds all edge, direction, and moulin data in a Point class
        self.data = np.empty(shape=self.DEM.shape,dtype=Point)
        p = Point(None,False)
        rows, cols = self.DEM.shape
        for i in range(0, rows):
            for j in range(0, cols):
                self.data[i][j] = p
        
        # Set edge and direction values of points on the edges 
        self.data[0,:] = Point(None,True) # Top side
        self.data[:,-1] = Point(None,True) # Right side
        self.data[-1,:] = Point(None,True) # Bottom side
        self.data[:,0] = Point(None,True) # Left side 

        # All_surfs holds all the surface melt rasters, which are the same size as the DEM and each cell represents the amount of melt (mm/day)
        # Assumes that the surface melt raster is applicable for a single day
        # All_surf_names hold the date in YYYYMMDD corresponding to each all_surf raster
        self.all_surfs = []
        self.all_surf_names = []
        for f in glob.glob(filepath_to_surfc):
            # Opens file, finds missing value, adds it to array, adds row of surface melt raster to array, and adds the date in YYYYMMDD of file to array
            src = gdal.Open(f)
            surf = np.array(src.GetRasterBand(1).ReadAsArray())
            surf = np.where(surf < 10, surf, 0)
            path = os.path.split(f)
            self.all_surfs.append(surf)
            self.all_surf_names.append(path[1][0:8])
        self.all_surfs = np.array(self.all_surfs)
        self.all_surf_names = np.array(self.all_surf_names)

        # Set up dictionary to convert month numbers into 3-letter abbreviations
        self.months = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
                       7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}



    def run(self, startdate, enddate):
        '''
        This is the main function that runs the model and takes a start date and end date as input to run. Dates in this range will exist
        in all_surf_names. The function is the shell that runs the model for the entire date range. For each day, it will find the correct
        surface melt raster, add it to the current water raster, and assuming the file for the final water raster doesn't exist yet, it will run the
        model and save the raster. If the final water raster does exist, it will load it and move onto the next day
        '''
        # All_waters holds all final water raster and is organized like this: [date][raster]
        self.all_waters = []
        self.all_moulins = []

        # Sets up list of days to run model for
        start_date = date(int(startdate[0:4]), int(startdate[4:6]), int(startdate[6:8]))
        end_date = date(int(enddate[0:4]), int(enddate[4:6]), int(enddate[6:8]))
        self.run_dates = pd.date_range(start_date, end_date)
        self.dates = pd.DataFrame(index=self.run_dates, data=np.arange(0, len(self.run_dates)), 
                                         columns=['index'])

        # Print staring model
        print('Starting model run: '+self.months[start_date.month]+' '+str(start_date.day)+', '+str(start_date.year)+
                              ' to '+self.months[end_date.month]+' '+str(end_date.day)+', '+str(end_date.year)+'...')

        # For each date in list of days to model
        for d in range(0,len(self.run_dates)):
            # Convert date to string format of all_surf_names
            day = self.run_dates[d]
            d_str = str(day.year)+str(day.month).zfill(2)+str(day.day).zfill(2)

            # Find surface melt raster that matches current date
            for i in range(0, len(self.all_surf_names)):
                if d_str == self.all_surf_names[i]:
                    break

            # The purpose of the dates dataframe is so that there is a corresponding date to each all_waters raster and so that given a date
            # one can find the appropriate index for the all_waters array
            # It is meant to be useful for graphing, so that a range of dates can be provided for graphs

            # Print date to be run
            print('  running day '+self.months[day.month]+' '+str(day.day)+', '+str(day.year))

            # Sets filepaths to files where final water raster and final moulin raster for this day will be saved
            curr_w = self.path_to_files+'Water_'+str(self.threshold)+'t_'+str(day.month)+'_'+str(day.day)+'_'+str(day.year)+'.npy'
            if (self.moulin_use):
                curr_w = self.path_to_files+'Water_'+str(self.threshold)+'t_'+str(self.moulin_thresh)+'m_'+str(day.month)+'_'+str(day.day)+'_'+str(day.year)+'.npy'
                curr_m = self.path_to_files+'Moulin_'+str(self.threshold)+'t_'+str(self.moulin_thresh)+'m_'+str(day.month)+'_'+str(day.day)+'_'+str(day.year)+'.npy'

            # If this file already exists, load it and set water to it 
            if (path.exists(curr_w)):
                self.water = np.load(curr_w)
                # If using moulins, load saved moulin file into moulin variable
                if (self.moulin_use):
                    self.moulin = np.load(curr_m)
            # Else, add the surface melt raster for that day (each cell is mm melt/day)
            # After running the model, save the finished water raster
            else:
                self.water += self.all_surfs[i]
                self.model_step()
                np.save(curr_w, self.water)
                if (self.moulin_use):
                    np.save(curr_m, self.moulin)
                    print(np.count_nonzero(self.moulin))


    def update_min_slope(self, slope_i, i):
        '''
        This function compares a newly calculated slope against the current smallest slope.
        If the new slope is smaller, min_s & min_i (minimum slope value & minimum slope index in slope_i)
        are updated.
        '''
        if (slope_i[i] < self.min_s):
            self.min_i = i
            self.min_s = slope_i[i]

    '''
    This next series of functions move water given a direction, row & column indices, and half the 
    elevation (water + DEM) difference between the center cell and the cell to which the water will be moved.
    If the amount water in the center cell (from which water will be removed) is equal to or less than that difference,
    all of the water is added to the surrounding cell.
    Otherwise, the difference amount will be removed from the center cell and added to the surrounding cell, while the 
    remainder of the water stays in the center cell.
    The reason the center cell operation is += is because the updated_ras (the new water iteration raster) starts out
    full of zeros.

    32 64 128
    16     1   0 == pool
    8   4  2   None == flow off edge
    '''

    def water_1(self, d, i, j):
        if self.water[i][j] <= d:
            self.updated_ras[i][j+1] += self.water[i][j]
        else:
            self.updated_ras[i][j+1] += d
            self.updated_ras[i][j] += self.water[i][j]-d

    def water_2(self, d, i, j):
        if self.water[i][j] <= d:
            self.updated_ras[i+1][j+1] += self.water[i][j]
        else:
            self.updated_ras[i+1][j+1] += d
            self.updated_ras[i][j] += self.water[i][j]-d

    def water_4(self, d, i, j):
        if self.water[i][j] <= d:
            self.updated_ras[i+1][j] += self.water[i][j]
        else:
            self.updated_ras[i+1][j] += d
            self.updated_ras[i][j] += self.water[i][j]-d

    def water_8(self, d, i, j):
        if self.water[i][j] <= d:
            self.updated_ras[i+1][j-1] += self.water[i][j]
        else:
            self.updated_ras[i+1][j-1] += d
            self.updated_ras[i][j] += self.water[i][j]-d

    def water_16(self, d, i, j):
        if self.water[i][j] <= d:
            self.updated_ras[i][j-1] += self.water[i][j]
        else:
            self.updated_ras[i][j-1] += d
            self.updated_ras[i][j] += self.water[i][j]-d

    def water_32(self, d, i, j):
        if self.water[i][j] <= d:
            self.updated_ras[i-1][j-1] += self.water[i][j]
        else:
            self.updated_ras[i-1][j-1] += d
            self.updated_ras[i][j] += self.water[i][j]-d

    def water_64(self, d, i, j):
        if self.water[i][j] <= d:
            self.updated_ras[i-1][j] += self.water[i][j]
        else:
            self.updated_ras[i-1][j] += d
            self.updated_ras[i][j] += self.water[i][j]-d

    def water_128(self, d, i, j):
        if self.water[i][j] <= d:
            self.updated_ras[i-1][j+1] += self.water[i][j]
        else:
            self.updated_ras[i-1][j+1] += d
            self.updated_ras[i][j] += self.water[i][j]-d

    
    def form_new_moulins(self):
        '''
        This function finds the all the new moulins after the model has finished iterating through a day. 
        It finds them by setting all cells with water depths over the given threshold to be moulins, then adding
        those new moulins to the moulin raster.
        '''
        
        new_moulins = self.water > self.moulin_thresh
        print(np.count_nonzero(new_moulins))
        self.moulin = np.logical_or(self.moulin, new_moulins)

    def model_step(self):
        '''
        This function runs the model for a single day, iterating over the day and moving the water once in each iteration.
        It stops once the maximum cell-cell difference between the previous and current iteration is less than a given threshold value.
        '''

        # The previous iteration water is set to be the water raster with the starting water from an all_surf raster and the water raster
        # from the previous day
        prev_water = self.water

        # Set a directions variable containing possible water flow directions
        directions = np.array([32,64,128,16,1,8,4,2])

        # Set dimensions for indexes
        l = len(self.indexes[0])

        # Runs until broken by the max difference being under the threshold
        while(True):
            # Initializes the surface used to calculate slopes as the DEM & the water
            self.water_dem = self.water + self.DEM

            # If using moulins, set any cell where there is a moulin to an elevation of 0, ensuring that water will always
            # drain down into this cell
            if (self.moulin_use):
                self.water_dem = np.where(self.moulin,0,self.water_dem)
        
            # This will contain the water raster after moving all the water once, initialized to be all zeros
            self.updated_ras = np.zeros(self.water.shape)

            # For all row, column pairs in DEM
            for k in range(0,l):
                # Set row, column values
                i = self.indexes[0][k]
                j = self.indexes[1][k]

                # Extract point data from data array
                point = self.data[i][j]

                # If point is on the edge, water flows off
                if (point.edge):
                    self.updated_ras[i][j] += 0.

                # If using moulins and the cell is a moulin, skip the cell because no water will be flowing out of it
                elif (self.moulin_use and self.moulin[i][j]):
                    continue

                # Else, point is not on the edge and the neighboring cell with the small slope needs to be found
                # for the water to move
                else:
                # Holds the slopes for an indexed cell with its surrounding cell
                # The key below shows where each index goes to (left)
                    slope_i = np.zeros(8)
                    '''
                    0 1 2   32 64 128
                    3   4   16     1   0 == pool
                    5 6 7   8   4  2   None == flow off edge
                    negative slope means sloping away from processed cell
                    positive slope means sloping towards processed cell
                    zero slope means same height as processed cell
                    None slope means that the cell to be compared with the indexed cell is off the model domain
                    '''
                    # Minimum slope & corresponding index to slope_i
                    self.min_s = np.inf
                    self.min_i = 0

                    # This section calculates all the slopes for the indexed cell with its neighboring cells, assuming it is not an edge cell
                    # Those slopes divided by sqrt(2) are corner cells
                    # Along the way it tries to find the smallest slope & corresponding index

                    # Slope direction 0
                    slope_i[0] = (self.water_dem[i-1][j-1]-self.water_dem[i][j])/np.sqrt(2)
                    self.update_min_slope(slope_i, 0)

                    # Slope direction 1
                    slope_i[1] = self.water_dem[i-1][j]-self.water_dem[i][j]
                    self.update_min_slope(slope_i, 1)

                    # Slope direction 2
                    slope_i[2] = (self.water_dem[i-1][j+1]-self.water_dem[i][j])/np.sqrt(2)
                    self.update_min_slope(slope_i, 2)

                    # Slope direction 3
                    slope_i[3] = self.water_dem[i][j-1]-self.water_dem[i][j]
                    self.update_min_slope(slope_i, 3)

                    # Slope direction 4
                    slope_i[4] = self.water_dem[i][j+1]-self.water_dem[i][j]
                    self.update_min_slope(slope_i, 4)

                    # Slope direction 5
                    slope_i[5] = (self.water_dem[i+1][j-1]-self.water_dem[i][j])/np.sqrt(2)
                    self.update_min_slope(slope_i, 5)

                    # Slope direction 6
                    slope_i[6] = self.water_dem[i+1][j]-self.water_dem[i][j]
                    self.update_min_slope(slope_i, 6)

                    # Slope direction 7
                    slope_i[7] = (self.water_dem[i+1][j+1]-self.water_dem[i][j])/np.sqrt(2)
                    self.update_min_slope(slope_i, 7)

                    # If the minimum slope is greater than or equal to zero (in a valley)
                    if self.min_s >= 0:
                        point.direction = 0
                    # Else there is a neighboring cell lower than the indexed cell and that is the direction the water will flow
                    # The neighboring cell lowest compared to the indexed cell gets all the water (i.e. the flow is not split)
                    else:
                        point.direction = directions[self.min_i]

                    # If the water is pooling, add the water from that cell to the updated_ras
                    if point.direction == 0:
                        self.updated_ras[i][j] += self.water[i][j]
                    # Else the water is not flowing off the model domain or pooling
                    else:
                        # If cell is flowing to a corner, multiply by sqrt(2) to get back the origial elevation difference between the two cells
                        # d is one half the distance between the two cells
                        if point.direction == 32 or point.direction == 128 or point.direction == 8 or point.direction == 1:
                            d = (np.abs(self.min_s)*np.sqrt(2))/2
                        else:
                            d = np.abs(self.min_s)/2
                        
                        # Runs through all possible directions and according calls the right function to move the water
                        if point.direction == 1:
                            self.water_1(d, i, j)
                        elif point.direction == 2:
                            self.water_2(d, i, j)
                        elif point.direction == 4:
                            self.water_4(d, i, j)
                        elif point.direction == 8:
                            self.water_8(d, i, j)
                        elif point.direction == 16:
                            self.water_16(d, i, j)
                        elif point.direction == 32:
                            self.water_32(d, i, j)
                        elif point.direction == 64:
                            self.water_64(d, i, j)
                        elif point.direction == 128:
                            self.water_128(d, i, j)

            # Set water raster to previous water & water to update_ras (water after one movement)
            prev_water = self.water            
            self.water = self.updated_ras

            # If using moulins, set all cells where there are moulins to have a water value of 0, showing that the water
            # has flowed down into the ice sheet away from the surface
            # After that, form new moulins
            if (self.moulin_use):
                self.water = np.where(self.moulin, 0, self.water)
                self.form_new_moulins()

            # Calculate the difference between the current and previous waters, then take the absolute value
            diff = prev_water-self.water
            diff = np.abs(diff)
            max_diff = np.amax(diff)
            print('  ', max_diff)

            # If the largest difference is less than or equal to the threshold break the loop and thus the day is done
            if (max_diff <= self.threshold):
                print('  ', np.amax(self.water))
                break
            


    '''
    These next four functions are the graphing functions and can only be used after the run() function has been called.
    Graphing options include 3D, 2D, slice (only takes a row), and subslice (takes a portion of a row)
    The graphs are not saved here, but in the main program
    '''

    def plot_model3D(self, date):
        # index = int(self.dates[date])
        # t_water_dem = self.DEM+self.all_waters[index]
        curr_w = self.path_to_files+'Water_'+str(self.threshold)+'t_'+str(date.month)+'_'+str(date.day)+'_'+str(date.year)+'.npy'
        if (self.moulin_use):
            curr_w = self.path_to_files+'Water_'+str(self.threshold)+'t_'+str(self.moulin_thresh)+'m_'+str(date.month)+'_'+str(date.day)+'_'+str(date.year)+'.npy'
        t_water_dem = np.load(curr_w)+self.DEM

        fig = plt.figure(1)
        ax = fig.gca(projection='3d')
        ax.plot_surface(self.long_2D,self.lat_2D,self.DEM,cmap='gist_gray',alpha=0.6)
        ax.plot_surface(self.long_2D,self.lat_2D,t_water_dem,cmap='Blues')
        plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f°'))
        plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f°'))
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.zlabel('Altitude (m)')
        plt.show()

    def plot_model2D(self, date):
        colors_water = [(0,0,0,0),'lightskyblue','deepskyblue','dodgerblue','royalblue','mediumblue']
        levels_water = [0, 1, 2, 5, 10, 15]
        curr_w = self.path_to_files+'Water_'+str(self.threshold)+'t_'+str(date.month)+'_'+str(date.day)+'_'+str(date.year)+'.npy'
        if (self.moulin_use):
            curr_w = self.path_to_files+'Water_'+str(self.threshold)+'t_'+str(self.moulin_thresh)+'m_'+str(date.month)+'_'+str(date.day)+'_'+str(date.year)+'.npy'
        water = np.load(curr_w)

        fig = plt.figure(3, figsize=(8, 5))
        ax = fig.add_subplot(1, 1, 1)

        cs = ax.contourf(self.long_2D_deg,self.lat_2D_deg,self.DEM,cmap='gist_gray', levels=30)
        cs2 = ax.contourf(self.long_2D_deg,self.lat_2D_deg,water,levels_water,colors=colors_water)

        cbar = plt.colorbar(cs, shrink=1)
        cbar.set_label('Altitude (m)')
        cbar2 = plt.colorbar(cs2, shrink=1)
        cbar2.set_label('Water height (m)')

        plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f°'))
        plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f°'))
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

    def plot_zoom_model2D(self, date, up_b, low_b, left_b, right_b):
        colors_water = [(0,0,0,0),'lightskyblue','deepskyblue','dodgerblue','royalblue','mediumblue']
        levels_water = [0, 1, 2, 5, 10, 15]
        curr_w = self.path_to_files+'Water_'+str(self.threshold)+'t_'+str(date.month)+'_'+str(date.day)+'_'+str(date.year)+'.npy'
        if (self.moulin_use):
            curr_w = self.path_to_files+'Water_'+str(self.threshold)+'t_'+str(self.moulin_thresh)+'m_'+str(date.month)+'_'+str(date.day)+'_'+str(date.year)+'.npy'
            curr_m = self.path_to_files+'Moulin_'+str(self.threshold)+'t_'+str(self.moulin_thresh)+'m_'+str(date.month)+'_'+str(date.day)+'_'+str(date.year)+'.npy'
        water = np.load(curr_w)

        fig = plt.figure(3, figsize=(8, 5))
        ax = fig.add_subplot(1, 1, 1)

        cs = ax.contourf(self.long_2D_deg[up_b:low_b,left_b:right_b],self.lat_2D_deg[up_b:low_b,left_b:right_b],self.DEM[up_b:low_b,left_b:right_b],cmap='gist_gray', levels=30, zorder=1)
        cs2 = ax.contourf(self.long_2D_deg[up_b:low_b,left_b:right_b],self.lat_2D_deg[up_b:low_b,left_b:right_b],water[up_b:low_b,left_b:right_b],levels_water,colors=colors_water, zorder=2)
        if (self.moulin_use):
            moulin = np.load(curr_m)
            moulins = np.logical_not(moulin)[up_b:low_b,left_b:right_b]

            # Apply mask
            moulins_masked = np.ma.array(np.ones(moulins.shape), mask=moulins)
            long_masked = np.ma.array(self.long_2D_deg[up_b:low_b,left_b:right_b], mask=moulins).flatten()
            lat_masked = np.ma.array(self.lat_2D_deg[up_b:low_b,left_b:right_b], mask=moulins).flatten()

            sc = ax.scatter(long_masked,lat_masked,s=20,c='r',zorder=3)

        cbar = plt.colorbar(cs, shrink=1)
        cbar.set_label('Altitude (m)')
        cbar2 = plt.colorbar(cs2, shrink=1)
        cbar2.set_label('Water height (m)')

        plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f°'))
        plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f°'))
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')    

    def plot_slice(self, date, row):
        curr_w = self.path_to_files+'Water_'+str(self.threshold)+'t_'+str(date.month)+'_'+str(date.day)+'_'+str(date.year)+'.npy'
        if (self.moulin_use):
            curr_w = self.path_to_files+'Water_'+str(self.threshold)+'t_'+str(self.moulin_thresh)+'m_'+str(date.month)+'_'+str(date.day)+'_'+str(date.year)+'.npy'
        water = np.load(curr_w)
        t_water_dem = self.DEM+water
        plt.figure(3)
        plt.plot(self.long_2D_deg[row], self.DEM[row],color='black')
        plt.fill_between(self.long_2D_deg[row],self.DEM[row],t_water_dem[row],color='blue')
        plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f°'))
        plt.xlabel('Longitude')
        plt.ylabel('Altitude plus water height (m)') 

    def plot_sub_slice(self, date, row, low_b, up_b):
        curr_w = self.path_to_files+'Water_'+str(self.threshold)+'t_'+str(date.month)+'_'+str(date.day)+'_'+str(date.year)+'.npy'
        if (self.moulin_use):
            curr_w = self.path_to_files+'Water_'+str(self.threshold)+'t_'+str(self.moulin_thresh)+'m_'+str(date.month)+'_'+str(date.day)+'_'+str(date.year)+'.npy'
        water = np.load(curr_w)
        t_water_dem = self.DEM+water
        plt.figure(3)
        plt.plot(self.long_2D_deg[row][low_b:up_b],self.DEM[row,low_b:up_b],color='black')
        plt.fill_between(self.long_2D_deg[row][low_b:up_b],self.DEM[row,low_b:up_b],t_water_dem[row,low_b:up_b],color='blue',alpha=0.2)
        plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f°'))
        plt.xlabel('Longitude')
        MAX = np.amax(self.DEM[row,low_b:up_b])
        MIN = np.amin(self.DEM[row,low_b:up_b])
        plt.ylim([MIN,MAX])
        plt.ylabel('Altitude plus water height (m)')






