import pickle
from sklearn.preprocessing import MinMaxScaler
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from Actor_Network import ActorNetwork

state_dim = 131
action_dim = 4
BATCH_SIZE = 32
TAU = 0.001
LRA = 0.001

sess = tf.Session()

K.set_session(sess)

actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)

ID = pd.read_csv("./Data_Bases/ID_val.csv")

actions = pd.read_csv("./Data_Bases/Action_data_val.csv")

states = pd.read_csv("./Data_Bases/States_data_val.csv")

state_scaler = pickle.load(open("./Models/state_scaler.pkl","rb"))

action_scaler = pickle.load(open("./Models/action_scaler.pkl","rb"))

kpi_model = keras.models.load_model("./Models/KPI_model_best.h5")

actor.model.load_weights("./Models/No_Noise_result/actormodel9999.h5")

imitation_model = keras.models.load_model("./Models/imitation_model.h5")

KPIs = pd.read_csv("./Data_Bases/KPI_data_val.csv")

cur_cell = " "

total_num = 0.0
improve_num_human = 0.0
improve_num_policy = 0.0
improve_num_imit = 0.0
total_improvement_human = 0.0
total_improvement_policy = 0.0
total_improvement_imit = 0.0

ini_kpi = 0


for i in range(ID.shape[0]):
    tmp_cell = ID.iloc[i,1]
    cur_state = state_scaler.transform(states.iloc[i,:].values.reshape(1,-1))
    human_action = action_scaler.transform(actions.iloc[i, :].values.reshape(1,-1))

    actor_action = actor.model.predict(state_scaler.transform(states.iloc[i, :].values.reshape(1,-1)))
    imitation_action = imitation_model.predict(state_scaler.transform(states.iloc[i, :].values.reshape(1,-1)))

    human_kpi = KPIs.iloc[i].values
    actor_kpi = kpi_model.predict([cur_state, actor_action])
    imitation_kpi = kpi_model.predict([cur_state, imitation_action])
    if tmp_cell != cur_cell:
        if ini_kpi != 0:
            if human_kpi < ini_kpi:
                improve_num_human += 1
                total_improvement_human += (ini_kpi - human_kpi)/ini_kpi * 100
            if actor_kpi < ini_kpi:
                improve_num_policy += 1
                total_improvement_policy += (ini_kpi - actor_kpi)/ini_kpi * 100
            if imitation_kpi < ini_kpi:
                improve_num_imit += 1
                total_improvement_imit += (ini_kpi - imitation_kpi)/ini_kpi * 100
        cur_cell = tmp_cell
        total_num += 1
        print("Now at cell: " + cur_cell)
        ini_kpi = kpi_model.predict([cur_state,human_action])

    if i == ID.shape[0] - 1:
        if human_kpi < ini_kpi:
            improve_num_human += 1
        if actor_kpi < ini_kpi:
            improve_num_policy += 1

        total_num += 1

print("The Policy is able to improve {} % cells".format(improve_num_policy/total_num * 100))

print("The Human Expert is able to improve {} % cells".format(improve_num_human/total_num * 100))

print("The imitation Learning is able to improve {} % cells".format(improve_num_imit/total_num * 100))

print("The average improvement of the human experts is {} %.".format(total_improvement_human/improve_num_human))

print("The average improvement of the RL agent is {} %.".format(total_improvement_policy/improve_num_policy))

print("The average improvement of the Imitation Learning is {} %.".format(total_improvement_imit/improve_num_imit))





