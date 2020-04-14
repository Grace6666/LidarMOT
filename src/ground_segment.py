import numpy as np

class GroundSeg(object):
    def __init__(self, sensor_height = 1.5, num_iter = 3, num_seed = 20, thres_seed = 1.2, thres_ground_dist = 0.2):
        self.sensor_height = sensor_height
        self.num_iter = num_iter
        self.num_seed = num_seed
        self.thres_seed = thres_seed
        self.thres_ground_dist = thres_ground_dist

    def _extract_initial_seeds(self, pc):
        new_pc = pc[np.argwhere(pc[:,2] > -1.5 * self.sensor_height), :].squeeze()
        cnt = np.minimum(self.num_seed, new_pc.shape[0])
        index = np.argpartition(new_pc[:,2], cnt)
        index = index[:cnt]
        seeds = new_pc[index, :]
        return seeds

    def _estimatePlane(self, ground_pc):
        # using least square method: ax+by+d = z: normal = (a, b, -1)
        G = np.ones((ground_pc.shape[0], 3))
        G[:, 0] = ground_pc[:, 0]  # X
        G[:, 1] = ground_pc[:, 1]  # Y
        Z = ground_pc[:, 2]
        (a, b, d), resid, rank, s = np.linalg.lstsq(G, Z, rcond=None)
        normal = np.array([a, b, -1])
        normal = normal[:, np.newaxis]
        return d, normal


    def segment(self, pc):
        seeds = self._extract_initial_seeds(pc)
        ground_pc = seeds
        for i in range(self.num_iter):
            d, normal = self._estimatePlane(ground_pc)
            distance = np.squeeze(np.absolute(np.dot(pc[:, :3], normal) + d)/np.linalg.norm(normal)) # dist = |ax+by-z+d|/sqrt(a^2+^2+(-1)^2)
            ground_idx = np.argwhere(distance <= self.thres_ground_dist).squeeze()
            non_ground_idx = np.argwhere(distance > self.thres_ground_dist).squeeze()
            ground_pc = pc[ground_idx, :]
            non_ground_pc = pc[non_ground_idx, :]
            self.d = d / np.linalg.norm(normal)
            self.normal = normal / np.linalg.norm(normal)
        return ground_pc, non_ground_pc

    def get_plane_params(self):
        return self.normal, self.d