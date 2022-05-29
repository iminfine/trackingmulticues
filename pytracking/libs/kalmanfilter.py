from filterpy.kalman import KalmanFilter
from scipy.linalg import block_diag
from filterpy.common import Q_discrete_white_noise
import numpy as np

def xywh2center(box):
    return [box[0]+box[2]/2,box[1]+box[3]/2,box[2],box[3]]

def center2xywh(box):
    return [box[0]-box[2]/2,box[1]-box[3]/2,box[2],box[3]]

def klfilter(box):
    tracker = KalmanFilter(dim_x=6, dim_z=4)
    dt = 1   # time step

    tracker.F = np.array([[1,0,0,0,dt,0],
                          [0,1,0,0,0,dt],
                          [0,0,1,0,0,0],
                          [0,0,0,1,0,0],
                          [0,0,0,0,1,0],
                          [0,0,0,0,0,1]])
    tracker.u = 0.
    tracker.H = np.array([[1,0,0,0,0,0],
                          [0,1,0,0,0,0],
                          [0,0,1,0,0,0],
                          [0,0,0,1,0,0]])

    tracker.R = np.eye(4) * 0.35**2
    q = Q_discrete_white_noise(dim=3, dt=dt, var=0.04**2)
    tracker.Q = block_diag(q, q)
    tracker.x = np.array([[box[0], box[1], box[2], box[3],0,0]]).T
    tracker.P = np.eye(6)
    return tracker

class motionfilter(object):
    def __init__(self, box):
        box=xywh2center(box)
        self.tracker=klfilter(box)
    def predict(self):
        self.tracker.predict()
        state=self.tracker.x
        pbox=center2xywh(state[:4].T[0])
        v=state[-2:].T[0]
        return pbox,v
    def update(self,box):
        box=xywh2center(box)
        return self.tracker.update(box)
    def transfer(self,delta):
        x=self.tracker.x
        x[0]=x[0]+delta[0]
        x[1]=x[1]+delta[1]
    def reset(self,box,v):
        box=xywh2center(box)
        dt=1
        self.tracker.x = np.array([[box[0], box[1], box[2], box[3],v[0],v[1]]]).T
        self.tracker.R = np.eye(4) * 0.35**2
        q = Q_discrete_white_noise(dim=3, dt=dt, var=0.04**2)
        self.tracker.Q = block_diag(q, q)
        self.tracker.P = np.eye(6)

