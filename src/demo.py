from utils import load_velo_scan, box2corners_2d
from utils_io import *

import os
import mayavi.mlab as mlab
import numpy as np

from roi_extractor import ROIExtractor
from ground_segment import GroundSeg
from detection import Detector
from tracking import MOTracker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)


def init_mlab_fig(fig):
    # plot origin and axis
    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.5)
    axes = np.array([
        [2., 0., 0., 0.],
        [0., 2., 0., 0.],
        [0., 0., 2., 0.],
    ], dtype=np.float64)
    mlab.plot3d([0, axes[0, 0]], [0, axes[0, 1]], [0, axes[0, 2]], color=(1, 0, 0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[1, 0]], [0, axes[1, 1]], [0, axes[1, 2]], color=(0, 1, 0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[2, 0]], [0, axes[2, 1]], [0, axes[2, 2]], color=(0, 0, 1), tube_radius=None, figure=fig)
    w = 50
    l = 60
    boundary = np.array([[-l / 2, -w / 2, 0],
                         [l / 2, -w / 2, 0],
                         [l / 2, w / 2, 0],
                         [-l / 2, w / 2, 0],
                         [-l / 2, -w / 2, 0]])
    mlab.plot3d(boundary[:,0], boundary[:,1], boundary[:,2], color=(0, 0, 1))
    mlab.view(azimuth=0, elevation=0, figure=fig)

def mlab_show(fig, gd_pc, ngd_pc_ = None, ngd_bb_2d = None, tracks = None, max_color = 30):
    mlab.clf()
    init_mlab_fig(fig)
    # plot ground points
    mlab.points3d(gd_pc[:, 0], gd_pc[:, 1], gd_pc[:, 2], color=(1, 1, 1), mode='point', colormap='gnuplot',
                  reset_zoom = False, scale_factor=1, figure=fig)

    # plot non ground points or in clusters and corresponding bouonding boxes
    colors = gen_colors(max_color)
    if ngd_pc_:
        if len(ngd_pc_) == 1:
            ngd_pc = ngd_pc_[0]
            mlab.points3d(ngd_pc[:, 0], ngd_pc[:, 1], ngd_pc[:, 2], ngd_pc[:, 3], mode='point', colormap='gnuplot',
                          reset_zoom = False, scale_factor=1, figure=fig)
        else:
            clusters_list, ngd_other = ngd_pc_
            mlab.points3d(ngd_other[:, 0], ngd_other[:, 1], ngd_other[:, 2], color=(1, 1, 1), mode='point', colormap='gnuplot',
                          reset_zoom = False, scale_factor=1, figure=fig)
            for i, cluster in enumerate(clusters_list):
                color_temp = colors[i % max_color]
                mlab.points3d(cluster[:, 0], cluster[:, 1], cluster[:, 2], color=color_temp, mode='point',
                              colormap='gnuplot', reset_zoom = False,
                              scale_factor=1, figure=fig)
    if not (ngd_bb_2d is None):
        for i in range(ngd_bb_2d.shape[0]):
            bb_2d = ngd_bb_2d[i, :]
            ngd_bb_corners_2d = box2corners_2d(bb_2d)
            temp = np.vstack((ngd_bb_corners_2d, ngd_bb_corners_2d[0, :][np.newaxis, :]))
            temp = np.hstack((temp, np.zeros((temp.shape[0], 1))))
            mlab.plot3d(temp[:, 0], temp[:, 1], temp[:, 2], color=(1, 1, 1), reset_zoom = False, figure=fig)

    if not (tracks is None):
        for i in range(tracks.shape[0]):
            bb_2d = tracks[i, :5]
            text = 'ID: %d' % tracks[i, 5]
            ngd_bb_corners_2d = box2corners_2d(bb_2d)
            color_temp = colors[i % max_color]
            temp = np.vstack((ngd_bb_corners_2d, ngd_bb_corners_2d[0, :][np.newaxis, :]))
            temp = np.hstack((temp, np.zeros((temp.shape[0], 1))))
            mlab.plot3d(temp[:, 0], temp[:, 1], temp[:, 2], color=color_temp, reset_zoom = False, figure=fig)
            mlab.text3d(temp[0, 0], temp[0, 1], temp[0, 2], text, scale=0.8, color=color_temp, figure=fig)



def multi_target_tracking(seq, figure=None, gen_dect = False):
    print('Sequence: %d' % (seq))
    velo_dir = os.path.join(ROOT_DIR, 'velodyne', '%04d' % seq)
    _, num_velo_files = load_list_from_folder(velo_dir)
    str_frame = 0
    num_frames = num_velo_files
    save_file_dir = os.path.join(ROOT_DIR, 'results', '%04d' % seq)
    save_img_dir = os.path.join(save_file_dir, 'img')
    mkdir_if_missing(save_file_dir)
    mkdir_if_missing(save_img_dir)
    # initial classes
    z_range = [5, -8]
    dist_range = [50, 2]
    roi_e = ROIExtractor(z_range[0], z_range[1], dist_range[0], dist_range[1])
    gd_seg = GroundSeg()
    ngc_detector = Detector()
    tracker = MOTracker()
    for frame in range(str_frame, str_frame+num_frames):
        print('frame: %d' % (frame))
        velo_file = os.path.join(velo_dir, '%06d.bin' % (frame))
        save_path = os.path.join(save_img_dir, '%06d.png' % (frame))
        dect_dir = os.path.join(save_file_dir, 'dect', '%06d' % (frame))
        mkdir_if_missing(dect_dir)
        gd_pc_file = os.path.join(dect_dir, 'gd_pc.npy')
        clusters_list_file = os.path.join(dect_dir, 'clusters_list.npy')
        ngd_other_file = os.path.join(dect_dir, 'ngd_other.npy')
        dects_file = os.path.join(dect_dir, 'dects.npy')
        flag = is_path_exists(gd_pc_file) & is_path_exists(clusters_list_file) & is_path_exists(
            ngd_other_file) & is_path_exists(dects_file)
        # generate / get detections at current frame
        if not flag or gen_dect:
            pc = load_velo_scan(velo_file)
            # region of interest extraction
            pc = roi_e.roi_extract(pc)
            # ground and non-ground segmentation
            gd_pc, ngd_pc = gd_seg.segment(pc)
            clusters_list, ngd_other = ngc_detector.clustering(ngd_pc, filter=True)
            dect_boxes_2d = ngc_detector.bounding_box_2d(clusters_list, criterion='variance')  # [(4x2)]
            np.save(gd_pc_file, gd_pc)
            np.save(clusters_list_file, clusters_list)
            np.save(ngd_other_file, ngd_other)
            np.save(dects_file, dect_boxes_2d)
        else:
            # load detection results
            gd_pc = np.load(gd_pc_file)
            clusters_list = np.load(clusters_list_file, allow_pickle=True)
            clusters_list = list(clusters_list)
            ngd_other = np.load(ngd_other_file)
            dect_boxes_2d = np.load(dects_file)

        # tracking
        tracks = tracker.update(dect_boxes_2d)

        if figure != None:
            fig = mlab_show(figure, gd_pc, ngd_pc_=[clusters_list, ngd_other], ngd_bb_2d=dect_boxes_2d, tracks = tracks)
            mlab.savefig(save_path, figure=fig)
            # mlab.show(10)


sequence = [6]
show = True
for seq in sequence:
    if show:
        fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(2000, 2000))
        multi_target_tracking(seq, figure=fig)
    else:
        multi_target_tracking(seq)




