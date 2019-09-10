import numpy as np
import random
import tensorflow as tf
from tensorflow.python.keras import backend as K


from Simulator import Val_Simulator
from Buffer import Buffer
from Actor_Network import ActorNetwork
from Critic_Network import CriticNetwork
from OU import OU
import time
from sklearn.metrics import mean_squared_error
OU = OU()  # Ornstein-Uhlenbeck Process


def playGame(train_indicator=1):  # 1 means Train, 0 means simply Run
    model_path = "./Models/"
    result_path = "./Results/"
    curr_test = "Large_Noise_Result/"

    BUFFER_SIZE = 10000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001  # Target Network HyperParameters
    LRA = 1e-4 # Learning rate for Actor
    LRC = 1e-3  # Lerning rate for Critic
    action_dim = 4  # Steering/Acceleration/Brake
    state_dim = 131  # of sensors input

    np.random.seed(2333)

    EXPLORE = 10000
    episode_count = 10000
    max_steps = 100000
    reward = 0
    done = 0
    step = 0
    epsilon = 1

    # Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    K.set_session(sess)

    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)

    buff = Buffer(BUFFER_SIZE)  # Create replay buffer

    # Generate a Torcs environment
    env = Val_Simulator()

    # Now load the weight
    for i in range(episode_count):
        start_time = time.time()
        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))
        if i % 1000 == 0:
            # losses = np.zeros((1000,))
            total_rewards = np.zeros((1000,))


        s_t = env.reset()

        total_reward = 0
        loss = 0
        for j in range(max_steps):
            epsilon -= 1.0 / EXPLORE

            a_t = actor.model.predict(s_t)
            a_t = np.around(a_t,decimals=1)

            s_t1, r_t, done= env.step(a_t)


            # buff.add(s_t, a_t,r_t, np.array([[done]]),s_t1)  # Add replay buffer
            #
            # # Do the batch update
            #
            # batch = buff.getBatch(BATCH_SIZE)
            # states = batch[:,:state_dim]
            # actions = batch[:,state_dim:state_dim+action_dim]
            # rewards = batch[:,state_dim+action_dim]
            # new_states = batch[:,state_dim+action_dim+2:]
            # dones = batch[:,state_dim+action_dim+1]
            # y_t = actions.copy()
            #
            # target_q_values = critic.target_model.predict([new_states, np.around(actor.target_model.predict(new_states),decimals=1)])
            #
            # for k in range(len(batch)):
            #     if dones[k]:
            #         y_t[k] = rewards[k]
            #     else:
            #         y_t[k] = rewards[k] + GAMMA * target_q_values[k]
            #
            # loss += critic.model.evaluate([states,actions],y_t,verbose=0)

            total_reward += r_t

            print("Episode", i, "Step", step, "Action", a_t, "Reward", r_t, "Loss", loss)

            step += 1
            if done:
                break

        total_rewards[i % 1000] = total_reward

        if np.mod((i+1), 1000) == 0:
                # losses_path = (result_path + curr_test + 'losses_val{}.txt').format(i)
                rewards_path = (result_path + curr_test + 'rewards_val{}.txt').format(i)
                # np.savetxt(losses_path,losses)
                np.savetxt(rewards_path,total_rewards)
                print("Now we load model")
                actor.model.load_weights((model_path+curr_test+"actormodel{}.h5").format(i))
                critic.model.load_weights((model_path+curr_test+"criticmodel{}.h5").format(i))
                actor.target_model.load_weights((model_path + curr_test + "actortarmodel{}.h5").format(i))
                critic.target_model.load_weights((model_path + curr_test + "crititarcmodel{}.h5").format(i))

        print("TOTAL REWARD @ " + str(i) + "-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("Took {} S".format(time.time() - start_time))
  # This is for shutting down TORCS
    print("Finish.")


if __name__ == "__main__":
    playGame()