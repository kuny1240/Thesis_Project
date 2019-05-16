import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras import utils as np_utils
from tensorflow.keras import optimizers
from Simulator import *


class Agent(object):

    def __init__(self, input_dim, output_dim, hidden_dims=[32, 32]):
        """The agent for single parameter tuning
        Args:
            input_dim (int): the dimension of state.
                Same as `env.observation_space.shape[0]`
            output_dim (int): the number of discrete actions
                Same as `env.action_space.n`
            hidden_dims (list): hidden dimensions
        Methods:
            private:
                __build_train_fn -> None
                    It creates a train function
                    It's similar to defining `train_op` in Tensorflow
                __build_network -> None
                    It create a base model
                    Its output is each action probability
            public:
                get_action(state) -> action
                fit(state, action, reward) -> None
        """

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.__build_network(input_dim, output_dim, hidden_dims)
        self.__build_train_fn()

    def __build_network(self, input_dim, output_dim, hidden_dims=[32, 32]):
        """Create a base network"""
        self.model = tf.keras.models.Sequential()
        self.model.add(layers.BatchNormalization(input_shape=self.input_dim))

        for h_dim in hidden_dims:
            self.model.add(layers.Dense(h_dim, activation='relu'))
            # self.model.add(layers.Dropout(0.25))

        # I set output to be 5 at first,(+5,+1,0,-1,-5)

        self.model.add(layers.Dense(output_dim, activation='softmax'))

    def __build_train_fn(self):
        """Create a train function
        It replaces `model.fit(X, y)` because we use the output of model and use it for training.
        For example, we need action placeholder
        called `action_one_hot` that stores, which action we took at state `s`.
        Hence, we can update the same action.
        This function will create
        `self.train_fn([state, action_one_hot, discount_reward])`
        which would train the model.
        """
        action_prob_placeholder = self.model.output
        action_onehot_placeholder = K.placeholder(shape=(None, self.output_dim),
                                                  name="action_onehot")
        discount_reward_placeholder = K.placeholder(shape=(None,),
                                                    name="discount_reward")

        action_prob = K.sum(action_prob_placeholder * action_onehot_placeholder, axis=1)
        log_action_prob = K.log(action_prob)

        loss = - log_action_prob * discount_reward_placeholder
        loss = K.mean(loss)
        print(loss)

        adam = optimizers.Adam()

        updates = adam.get_updates(params=self.model.trainable_weights,
                                   loss=loss)

        self.train_fn = K.function(inputs=[self.model.input,
                                           action_onehot_placeholder,
                                           discount_reward_placeholder],
                                   outputs=[],
                                   updates=updates)

    def get_action(self, state):
        """Returns an action at given `state`
        Args:
            state (1-D or 2-D Array): It can be either 1-D array of shape (state_dimension, )
                or 2-D array shape of (n_samples, state_dimension)
        Returns:
            action: an integer action value ranging from 0 to (n_actions - 1)
        """
        action_prob = np.squeeze(self.model.predict(state))
        return np.random.choice(np.arange(self.output_dim), p=action_prob)

    def fit(self, S, A, R):
        """Train a network
        Args:
            S (2-D Array): `state` array of shape (n_samples, state_dimension)
            A (1-D Array): `action` array of shape (n_samples,)
                It's simply a list of int that stores which actions the agent chose
            R (1-D Array): `reward` array of shape (n_samples,)
                A reward is given after each action.
        """
        action_onehot = np_utils.to_categorical(A, num_classes=self.output_dim)
        discount_reward = compute_discounted_R(R)
        S = S.reshape((-1, 35))
        discount_reward = discount_reward.reshape((-1, 1))
        action_onehot = action_onehot.reshape((-1,11))

        self.train_fn([S, action_onehot, discount_reward])


def compute_discounted_R(R, discount_rate=.99):
    """Returns discounted rewards
    Args:
        R (1-D array): a list of `reward` at each time step
        discount_rate (float): Will discount the future value by this rate
    Returns:
        discounted_r (1-D array): same shape as input `R`
            but the values are discounted
    """
    discounted_r = np.zeros_like(R, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(R))):
        running_add = running_add * discount_rate + R[t]
        discounted_r[t] = running_add

    discounted_r -= discounted_r.mean() / discounted_r.std()

    return discounted_r


def run_episode(env, agent):
    """Returns an episode reward
    (1) Play until the game is done
    (2) The agent will choose an action according to the policy
    (3) When it's done, it will train from the game play
    Args:
        env (gym.env): Gym environment
        agent (Agent): Game Playing Agent
    Returns:
        total_reward (int): total reward earned during the whole episode
    """
    #     done = False
    S = []
    A = []
    R = []

    s = env.reset()

    total_reward = 0

    for i in range(21):
        a = agent.get_action(s)
        S.append(s)
        A.append(a)
        s2, r = env.step(a)
        total_reward += r

        R.append(r)

        s = s2

    print(total_reward)
    print(S)

    S = np.array(S)
    A = np.array(A)
    R = np.array(R)

    agent.fit(S, A, R)

    return total_reward


def main():
    output_dim = 11
    input_dim = [35]
    agent = Agent(input_dim, output_dim, [16, 16])
    env = Simulator()
    for i in range(10):
        reward = run_episode(env, agent)
        print(reward)


if __name__ == '__main__':
    main()