from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

class Detector(object):
    def __init__(self, eps = 0.45, min_samples = 10, leaf_size = 30, dtheta = 2.0):
        self.eps = eps
        self.min_samples = min_samples
        self.leaf_size = leaf_size
        self.dtheta = np.deg2rad(dtheta)  # delta theta for searching box direction
        self.min_dist_of_closeness_crit = 0.01 # parameter used in closeness criterion

    def clustering(self, pc, num_dim = 3, filter = True):
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples, leaf_size=self.leaf_size)
        pc_clusters = dbscan.fit(pc[:,:num_dim])
        pc_clusters_list = []
        pc_ngd = np.empty((0, pc.shape[1]))
        for i in set(pc_clusters.labels_):
            if i == -1: # -1 label means noise
                continue
            pc_cluster = pc[pc_clusters.labels_ == i]
            if filter:
                keep = self._cluster_filter(pc_cluster[:, :num_dim])
                if keep:
                    pc_clusters_list.append(pc_cluster)
                else:
                    pc_ngd = np.vstack((pc_ngd, pc_cluster))
            else:
                pc_clusters_list.append(pc_cluster)
        return pc_clusters_list, pc_ngd

    def _cluster_filter(self, pc_cluster):
        out = True
        xyz_range = np.max(pc_cluster, axis=0) - np.min(pc_cluster, axis=0)
        if xyz_range[0] < 1.5 and xyz_range[1] < 1.5:
            out = False
        if xyz_range[0]< 0.5 or xyz_range[1] < 0.5:
            out = False
        if xyz_range[0] > 10 or xyz_range[1] > 10 or xyz_range[2] > 10:
            out = False
        dist = np.min(np.abs(pc_cluster[:, 1]))
        if dist > 15:
            out = False
        return out

    def bounding_box_2d(self, pc_clusters_list, criterion = 'variance', filter = True):
        dect_boxes_2d = []
        for c in range(len(pc_clusters_list)):
            pc_cluster = pc_clusters_list[c]
            # search range
            max_theta = (-float('inf'), None)
            for theta in np.arange(0.0, np.pi/2.0 - self.dtheta, self.dtheta):
                e1 = np.array([np.cos(theta), np.sin(theta)]).reshape((-1, 1))
                e2 = np.array([-np.sin(theta), np.cos(theta)]).reshape((-1, 1))
                c1 = pc_cluster[:, :2] @ e1
                c2 = pc_cluster[:, :2] @ e2
                if criterion == 'area':
                    score = self._calc_area_criterion(c1, c2)
                elif criterion == 'closeness':
                    score = self._calc_closeness_criterion(c1, c2)
                elif criterion == 'variance':
                    score = self._calc_variance_criterion(c1, c2)

                if max_theta[0] < score:
                    max_theta = (score, theta)
            sin_s = np.sin(max_theta[1])
            cos_s = np.cos(max_theta[1])
            c1_s = pc_cluster[:, :2] @ np.array([cos_s, sin_s]).T
            c2_s = pc_cluster[:, :2] @ np.array([-sin_s, cos_s]).T
            c1_s_min = min(c1_s)
            c1_s_max = max(c1_s)
            c2_s_min = min(c2_s)
            c2_s_max = max(c2_s)
            box_2d = self._calc_bb(max_theta[1], c1_s_min, c1_s_max, c2_s_min, c2_s_max)
            if filter:
                out = self._box_filter(box_2d)
                if out:
                    dect_boxes_2d.append(box_2d)
            else:
                dect_boxes_2d.append(box_2d)
        dect_boxes_2d = np.array(dect_boxes_2d)
        return dect_boxes_2d

    def _box_filter(self, box_2d):
        out = True
        if box_2d[2] > 9 or box_2d[3] > 6:
            out = False
        return out

    def _calc_bb(self, theta, c1_s_min, c1_s_max, c2_s_min, c2_s_max):
        sin_s = np.sin(theta)
        cos_s = np.cos(theta)
        x = (sin_s * - (c2_s_min + c2_s_max) / 2 - cos_s * - (c1_s_min + c1_s_max) / 2)
        y = (-sin_s * - (c1_s_min + c1_s_max) / 2 - cos_s * - (c2_s_min + c2_s_max) / 2)
        w = c2_s_max - c2_s_min
        l = c1_s_max - c1_s_min
        if w > l:
            w, l = l, w
            theta = theta - np.pi/2
        box = np.array([x, y, l, w, theta])
        return box

    def _calc_bb_corners(self, theta, c1_s_min, c1_s_max, c2_s_min, c2_s_max):
        sin_s = np.sin(theta)
        cos_s = np.cos(theta)
        # four boundaries of a rect: [a,b,c]: ax+by-c=0
        rect = np.array([[cos_s, sin_s, c1_s_min],
                         [-sin_s, cos_s, c2_s_min],
                         [cos_s, sin_s, c1_s_max],
                         [-sin_s, cos_s, c2_s_max]])
        corners_2d = np.empty((4, 2))
        for i in range(3):
            a0, b0, c0 = rect[i, :]
            a1, b1, c1 = rect[i + 1, :]
            corners_2d[i, 0] = (b0 * -c1 - b1 * -c0) / (a0 * b1 - a1 * b0)  # x
            corners_2d[i, 1] = (a1 * -c0 - a0 * -c1) / (a0 * b1 - a1 * b0)  # y
        a0, b0, c0 = rect[3, :]
        a1, b1, c1 = rect[0, :]
        corners_2d[3, 0] = (b0 * -c1 - b1 * -c0) / (a0 * b1 - a1 * b0)  # x
        corners_2d[3, 1] = (a1 * -c0 - a0 * -c1) / (a0 * b1 - a1 * b0)  # y
        return corners_2d

    def _calc_area_criterion(self, c1, c2):
        c1_max = max(c1)
        c2_max = max(c2)
        c1_min = min(c1)
        c2_min = min(c2)
        alpha = -(c1_max - c1_min) * (c2_max - c2_min)
        return alpha

    def _calc_closeness_criterion(self, c1, c2):
        c1_max = max(c1)
        c2_max = max(c2)
        c1_min = min(c1)
        c2_min = min(c2)
        D1 = [min([np.linalg.norm(c1_max - ic1),
                   np.linalg.norm(ic1 - c1_min)]) for ic1 in c1]
        D2 = [min([np.linalg.norm(c2_max - ic2),
                   np.linalg.norm(ic2 - c2_min)]) for ic2 in c2]
        beta = 0
        for i, _ in enumerate(D1):
            d = max(min([D1[i], D2[i]]), self.min_dist_of_closeness_crit)
            beta += (1.0 / d)
        return beta

    def _calc_variance_criterion(self, c1, c2):
        c1_max = max(c1)
        c2_max = max(c2)
        c1_min = min(c1)
        c2_min = min(c2)
        D1 = [min([np.linalg.norm(c1_max - ic1),
                   np.linalg.norm(ic1 - c1_min)]) for ic1 in c1]
        D2 = [min([np.linalg.norm(c2_max - ic2),
                   np.linalg.norm(ic2 - c2_min)]) for ic2 in c2]
        E1, E2 = [], []
        for (d1, d2) in zip(D1, D2):
            if d1 < d2:
                E1.append(d1)
            else:
                E2.append(d2)
        V1 = 0.0
        if E1:
            V1 = - np.var(E1)
        V2 = 0.0
        if E2:
            V2 = - np.var(E2)
        gamma = V1 + V2
        return gamma