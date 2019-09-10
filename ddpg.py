import numpy as np
import random
import tensorflow as tf
from tensorflow.python.keras import backend as K


from Simulator import Simulator
from Buffer import Buffer
from Actor_Network import ActorNetwork
from Critic_Network import CriticNetwork
from OU import OU
import time
import os
OU = OU()  # Ornstein-Uhlenbeck Process


def playGame(train_indicator=1): # 1 means Train, 0 means simply Run

    cur_path = os.path.abspath(os.path.curdir)
    model_path = "/Models/"
    result_path = "/Results/"
    curr_test = "Large_Noise_Result/"
    actor_name = "actormodel{}.h5"
    critic_name = "criticmodel{}.h5"

    BUFFER_SIZE = 100000
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
    env = Simulator()

    # Now load the weight
    print("Now we load the weight")
    try:
        actor.model.load_weights(cur_path + "/Models/actormodel.h5")
        critic.model.load_weights(cur_path +"/Models/criticmodel.h5")
        actor.target_model.load_weights(cur_path + "/Models/actormodel.h5")
        critic.target_model.load_weights(cur_path + "/Models/criticmodel.h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")

    for i in range(episode_count):
        start_time = time.time()

        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))
        if i % 1000 == 0:
            losses = np.zeros((1000,))
            total_rewards = np.zeros((1000,))


        s_t = env.reset()

        total_reward = 0
        loss = 0
        for j in range(max_steps):
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1, action_dim])
            noise_t = np.zeros([1, action_dim])

            a_t_original = actor.model.predict(s_t)
            noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0],0.5, 1.00, 0.15)
            noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1], 0.5, 1.00, 0.15)
            noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], 0.5, 1.00, 0.15)
            noise_t[0][3] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][3], 0.5, 1.00, 0.15)

            # The following code do the stochastic brake
            # if random.random() <= 0.1:
            #    print("********Now we apply the brake***********")
            #    noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2],  0.2 , 1.00, 0.10)

            a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
            a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
            a_t[0][2] = a_t_original[0][2] + noise_t[0][2]
            a_t[0][3] = a_t_original[0][3] + noise_t[0][3]

            a_t = np.around(a_t,decimals=1)

            s_t1, r_t, done= env.step(a_t)


            buff.add(s_t, a_t,r_t, np.array([[done]]),s_t1)  # Add replay buffer

            # Do the batch update

            batch = buff.getBatch(BATCH_SIZE)
            states = batch[:,:state_dim]
            actions = batch[:,state_dim:state_dim+action_dim]
            rewards = batch[:,state_dim+action_dim]
            new_states = batch[:,state_dim+action_dim+2:]
            dones = batch[:,state_dim+action_dim+1]
            y_t = actions.copy()

            target_q_values = critic.target_model.predict([new_states, np.around(actor.target_model.predict(new_states),decimals=1)])

            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA * target_q_values[k]

            if (train_indicator):
                loss += critic.model.train_on_batch([states, actions], y_t)
                a_for_grad = np.around(actor.model.predict(states),decimals=1)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()


            total_reward += r_t
            s_t = s_t1

            print("Episode", i, "Step", step, "Action", a_t, "Reward", r_t, "Loss", loss)

            step += 1
            if done:
                break

        losses[i % 1000] = loss
        total_rewards[i % 1000] = total_reward

        if np.mod((i+1),100) == 0:
            if (train_indicator):
                print("Now we save model")
                actor.model.save_weights(cur_path + "/Models/actormodel.h5", overwrite=True)
                critic.model.save_weights(cur_path + "/Models/criticmodel.h5", overwrite=True)

        if np.mod((i+1), 1000) == 0:
            if (train_indicator):
                losses_path = (cur_path + result_path + curr_test + 'losses{}.txt').format(i)
                rewards_path = (cur_path + result_path + curr_test + 'rewards{}.txt').format(i)
                np.savetxt(losses_path,losses)
                np.savetxt(rewards_path,total_rewards)
                print("Now we save model")
                actor.model.save_weights((cur_path + model_path+curr_test+"actormodel{}.h5").format(i), overwrite=True)
                critic.model.save_weights((cur_path + model_path+curr_test+"criticmodel{}.h5").format(i), overwrite=True)
                actor.target_model.save_weights((cur_path + model_path + curr_test + "actortarmodel{}.h5").format(i), overwrite=True)
                critic.target_model.save_weights((cur_path + model_path + curr_test + "crititarcmodel{}.h5").format(i), overwrite=True)

        print("TOTAL REWARD @ " + str(i) + "-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("Took {} S".format(time.time() - start_time))

    # This is for shutting down TORCS
    print("Finish.")


if __name__ == "__main__":
    playGame()
