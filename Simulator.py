import numpy as np
import pandas as pd
import tensorflow as tf



class Simulator(object):
    def __init__(self):
        self.model = tf.keras.models.load_model('./Processed_Data/kpi_model_v2.h5')
        self.pos = 0
        self.state = np.array(35 * [0])
        self.act = 18  # this is the action column
        self.kpi = 3
        self.stationary = [17, 19, 20, 21, 22, 23, 24, 25, 26]  # these cols won't change during the procedure
        self.rand = list(range(1, 16)) + list(
            range(27, 37))  # these cols change a random gaussian noise with mean as the beginning status
        self.rand.remove(3)
        self.noise_base = np.array([0] * 24)
        self.action_space = np.array([40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140])
        self.cur_kpi = 0

    def reset(self):
        self.pos = int(np.round(np.random.rand(1)[0] * 1883614))
        self.state = pd.read_csv('./Processed_Data/simulator_data_big1.csv',
                                 index_col=None)[self.pos:self.pos + 1].values
        # basic possible states from existing points
        self.noise_base = self.state[:, self.rand] * 0.05  # at most 5% change
        self.cur_kpi = self.state[:, self.kpi]
        return self.state[:, list(range(1, 3)) + list(range(4, 37))]

    def step(self, action):
        self.state[:, self.act] = self.action_space[action]
        k = np.random.binomial(1, .5, len(self.rand))
        k[k == 0] = -1
        self.state[:, self.rand] = self.noise_base * np.random.rand(len(self.rand)) * k + self.state[:, self.rand]
        pred_kpi = self.model.predict(self.state[:, list(range(1, 3)) + list(range(4, 37))])
        r = pred_kpi - self.cur_kpi
        self.cur_kpi = pred_kpi

        return self.state[:, list(range(1, 3)) + list(range(4, 37))], r