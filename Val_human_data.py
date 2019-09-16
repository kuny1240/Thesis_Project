import numpy as np
import random
import tensorflow as tf
from tensorflow.python.keras import backend as K
import pandas as pd


from Simulator import Val_Simulator
from Buffer import Buffer
from Actor_Network import ActorNetwork
from Critic_Network import CriticNetwork
from OU import OU
import time


def val_human():
    # load data:
    training_states = pd.read_csv("./Data_Bases/States_data.csv",index_col=None)
    val_states = pd.read_csv("./Data_Bases/States_data_val.csv", index_col=None)
    training_kpi = pd.read_csv("./Data_Bases/KPI_data.csv",index_col=None)
    val_kpi = pd.read_csv("./Data_Bases/KPI_data_val.csv", index_col=None)
    training_label = pd.read_csv("./Data_Bases/ID.csv",index_col=None)
    val_label = pd.read_csv("./Data_Bases/ID_val.csv")

    # model load:
    model_path = "./Models/"
    result_path = "./Results/"
    test_set = ["No_Noise_Result/","Small_Noise_Result","Large_Noise_Result"]

    BUFFER_SIZE = 10000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001  # Target Network HyperParameters
    LRA = 1e-4  # Learning rate for Actor
    LRC = 1e-3  # Lerning rate for Critic
    action_dim = 4  # Steering/Acceleration/Brake
    state_dim = 131  # of sensors input

    np.random.seed(2333)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    K.set_session(sess)

    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)

    buff = Buffer(BUFFER_SIZE)  # Create replay buffer

    # Generate a Torcs environment
    env = Val_Simulator()


    for t in test_set:
        total_cell_training = 0
        total_cell_val = 0
        improved_cell_training_policy = 0
        improved_cell_training_human = 0
        improved_cell_val_policy = 0
        improved_cell_val_human = 0
        improved_kpi_training_policy = 0
        improved_kpi_training_human = 0
        improved_kpi_val_policy = 0
        improved_kpi_val_human = 0
        worst_training_policy = 0
        worst_val_policy = 0
        worst_training_human = 0
        worst_val_human = 0
        training_size = training_label.shape[0]
        for n in test_set:
            actor.model.load_weights((model_path + n + "actormodel9999.h5").format(i))
            actor.target_model.load_weights((model_path + n + "actortarmodel9999.h5").format(i))
            for i in range(training_size):
                if i == 0:
                    cur_cell = training_label[i,2]
                    pre_cell = None
                else:
                    cur_cell = training_label[i,2]
                if cur_cell == pre_cell:





