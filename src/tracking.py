import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
from utils import iou_2d, box2corners_2d
from filterpy.kalman import KalmanFilter

class KalmanBoxTracker(object):
    count = 0  # the number of established tracks in total
    def __init__(self, bbox2d):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        # x = [x, y, l, w, vx, vy].^T
        # z = [x, y, l, w].^T
        self.kf = KalmanFilter(dim_x=6, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0],  # state transition matrix
                              [0, 1, 0, 0, 0, 1],
                              [0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0],  # measurement function,
                              [0, 1, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0]])

        self.kf.Q[4:, 4:] *= 0.01  # process uncertainty/noise
        self.kf.x[:4] = bbox2d[:4].reshape((4, 1))  # velocities are initialized as zeros
        self.kf.P[4:,4:] *= 1000.  # state uncertainty, give high uncertainty to the unobservable initial velocities, covariance matrix
        self.kf.P *= 10.
        self.theta = bbox2d[4]


        self.frames_since_last_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 1           # number of total hits (the frames updated with det) including the first detection
        self.hit_streak = 1     # number of continuing hit considering the first detection
        self.first_continuing_hit = 1   # number of continuing hit in the first time
        self.still_first = True         # assign False when it is not associate with a detection
        self.age = 0            # number of frames since it is initialized, initial = 0

    def update(self, bbox2d):
        """
        Updates the state vector with observed bbox.
        bbox3D = [x, y, l, w, theta].^T
        """
        self.frames_since_last_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1  # number of continuing hit
        if self.still_first:
            self.first_continuing_hit += 1  # number of continuing hit in the first time

        self.kf.update(bbox2d[:4])
        self.theta = bbox2d[4]


    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        self.kf.predict()
        self.age += 1
        if (self.frames_since_last_update > 0):
            self.hit_streak = 0
            self.still_first = False
        self.frames_since_last_update += 1
        self.history.append(self.kf.x)
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        state = np.vstack((self.kf.x[:4], self.theta))
        state = state.reshape((5,))
        return state


class association(object):
    def __init__(self, method, iou_threshold=0.01):
        self.method = method
        self.iou_threshold = iou_threshold

    def associate_detections_to_trackers(self, detections, tracks):
        """
        detections:  N x 4 x 2
        trackers:    M x 4 x 2
        Returns 3 lists of matches, unmatched_detections and unmatched_trackers
        """
        if self.method == '2Dassignment':
            matches, unmatched_detections, unmatched_trackers = self.assignment(detections, tracks)
        return matches, unmatched_detections, unmatched_trackers

    def assignment(self, detections, tracks):
        if (len(tracks) == 0):
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 8, 3), dtype=int)
        iou_matrix = np.zeros((len(detections), len(tracks)), dtype=np.float32)

        for d, det in enumerate(detections):
            for t, trk in enumerate(tracks):
                iou_matrix[d, t] = iou_2d(det, trk)  # det: 4 x 2, trk: 4 x 2
        matched_indices = linear_assignment(-iou_matrix)  # hougarian algorithm

        unmatched_detections = []
        for d, det in enumerate(detections):
            if (d not in matched_indices[:, 0]):
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t, trk in enumerate(tracks):
            if (t not in matched_indices[:, 1]):
                unmatched_trackers.append(t)

        # filter out matched with low IOU
        matches = []
        for m in matched_indices:
            if (iou_matrix[m[0], m[1]] < self.iou_threshold):
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        if (len(matches) == 0):
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class MOTracker(object):
    def __init__(self, max_no_det_frames=2, min_hits=3, data_asso_method='2Dassignment'):
        # max age will preserve the bbox does not appear no more than 2 frames, interpolate the detection

        self.max_no_det_frames = max_no_det_frames   # maximum no detection associated frames - conf_track ->  terminate track
        self.min_hits = min_hits               # minimum frames associated with detection: tentative track -> conf_track
        self.association = association(data_asso_method)
        self.tracks = []       # collect all tracks
        self.frame_count = 0

    def update(self, dets):
        self.frame_count += 1
        ###### track prediction #######################################
        trks = np.zeros((len(self.tracks), 5))  # N x 5, #get predicted locations from existing trackers.
        to_del = []
        for t, trk in enumerate(trks):
            state = self.tracks[t].predict().reshape((-1, 1))
            trk[:] = [state[0], state[1], state[2], state[3], self.tracks[t].theta]
            if (np.any(np.isnan(state))):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.tracks.pop(t)

        ##### apply data association ##################################
        # dets_4corner:  N x 4 x 2
        # trks_4corner:  M x 4 x 2
        dets_4corner = [box2corners_2d(det_tmp) for det_tmp in dets]
        if len(dets_4corner) > 0:
            dets_4corner = np.stack(dets_4corner, axis=0)
        else:
            dets_4corner = []

        trks_4corner = [box2corners_2d(trk_tmp) for trk_tmp in trks]
        if len(trks_4corner) > 0:
            trks_4corner = np.stack(trks_4corner, axis=0)
        else:
            trks_4corner = []

        matched, unmatched_dets, unmatched_trks = self.association.associate_detections_to_trackers(dets_4corner, trks_4corner)

        ##### track update #############################################
        # update matched trackers with assigned detections
        for t, trk in enumerate(self.tracks):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]  # a list of index
                trk.update(dets[d, :][0])

        # track management
        # create and initialise new tracks for unmatched detections
        for i in unmatched_dets:  # a scalar of index
            trk = KalmanBoxTracker(dets[i, :])
            self.tracks.append(trk)

        # return confirmed tracks and delete tentative tracks
        i = len(self.tracks)
        report_tracks = []
        for trk in reversed(self.tracks):
            d = trk.get_state()  # bbox location
            if ((trk.frames_since_last_update < self.max_no_det_frames) and (
                    trk.hits >= self.min_hits )): #or self.frame_count <= self.min_hits
                report_tracks.append(
                    np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # remove tentative tracks
            if (trk.frames_since_last_update >= self.max_no_det_frames):
                self.tracks.pop(i)

        if (len(report_tracks) > 0):
            return np.concatenate(report_tracks)

        return np.empty((0, 6))