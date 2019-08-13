from tensorflow.python.keras.layers import Dense, Input, Add
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras import backend as K
import tensorflow as tf

HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 300


class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size

        K.set_session(sess)

        # Now create the model
        self.model, self.action, self.state = self.create_critic_network(state_size, action_size)
        self.target_model, self.target_action, self.target_state = self.create_critic_network(state_size, action_size)
        self.action_grads = tf.gradients(self.model.output, self.action)  # GRADIENTS for policy update
        self.sess.run(tf.initialize_all_variables())

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU) * critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, state_size, action_dim):
        print("Now we build the model")
        S = Input(shape=[state_size])
        A = Input(shape=[action_dim], name='action2')
        w1 = Dense(HIDDEN1_UNITS, activation='relu',kernel_initializer="glorot_normal")(S)
        a1 = Dense(HIDDEN2_UNITS, activation='linear',kernel_initializer="glorot_normal")(A)
        h1 = Dense(HIDDEN2_UNITS, activation='linear',kernel_initializer="glorot_normal")(w1)
        h2 = Add()([h1,a1])
        h3 = Dense(HIDDEN2_UNITS, activation='relu',kernel_initializer="glorot_normal")(h2)
        V = Dense(action_dim, activation='linear',kernel_initializer="glorot_normal")(h3)
        model = Model(inputs = [S, A], outputs = V)
        adam = Adam(lr=self.LEARNING_RATE,decay=1e-6)
        model.compile(loss='mse', optimizer=adam)
        return model, A, S