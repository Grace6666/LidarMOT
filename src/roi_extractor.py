"""
extract region of interest
"""


import numpy as np

class ROIExtractor(object):
    def __init__(self, z_max, z_min, r_max, r_min):
        self.z_max = z_max
        self.z_min = z_min
        self.r_max = r_max
        self.r_min = r_min
    
    def roi_extract(self, pc, elevation = True, planar_range = True):
        if elevation:
            pc = self.elevation_filter(pc)
        if planar_range:
            pc = self.planar_range_filter(pc)
        return pc

    def elevation_filter(self, pc):
        """
        filter out points located outsize the elevation range (z_min, z_max)
        lidar located at origin, pc in velo coordinate
        """
        new_pc = pc[(pc[:,2] >= self.z_min) & (pc[:,2] <= self.z_max),:]
        # print(pc[:,2].max(), pc[:,2].min())
        return new_pc

    def planar_range_filter(self, pc):
        """
        filter out points don't satisfy: r_min < sqrt(x^2 + y^2) < r_max
        """
        range2 = np.sqrt(np.square(pc[:,0]) + np.square(pc[:,1]))
        # print(np.max(range2), np.min(range2))
        new_pc = pc[((range2 < self.r_max) & (range2 > self.r_min)),:]
        return new_pc

    def voxelGridFilter(self, pc):
        """
        down sampling pc
        to do
        """
        pass
