import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import os

from tensorflow.python.keras.layers import Dense, Input, Add
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras import backend as K

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 400

def create_imitation_net(input_size, output_size, learning_rate):
    print("Now we build the model")
    S = Input(shape=[input_size])
    w1 = Dense(HIDDEN1_UNITS, activation='relu', kernel_initializer="glorot_normal")(S)
    h1 = Dense(HIDDEN2_UNITS, activation='relu', kernel_initializer="glorot_normal")(w1)
    V = Dense(output_size, activation='linear', kernel_initializer="glorot_normal")(h1)
    model = Model(inputs=S, outputs=V)
    adam = Adam(lr=learning_rate, decay=1e-6)
    model.compile(loss='mse', optimizer=adam)
    return model

if __name__ == "__main__":
    cur_path = os.path.abspath(os.path.curdir)
    state_scaler = pickle.load(open(cur_path + '/Models/state_scaler.pkl', 'rb'))
    action_scaler = pickle.load(open(cur_path + '/Models/action_scaler.pkl', 'rb'))

    states = pd.read_csv(cur_path + '/Data_Bases/States_data.csv',
                              index_col=None).values
    actions = pd.read_csv(cur_path + '/Data_Bases/Action_data.csv',
                               index_col=None).values

    X = state_scaler.transform(states)
    y = action_scaler.transform(actions)

    model = create_imitation_net(input_size=X.shape[1],output_size=y.shape[1],learning_rate=1e-3)

    model.fit(X,y,batch_size=32,epochs=100,validation_split=0.2)

    model.save("./Models/imitation_model.h5")