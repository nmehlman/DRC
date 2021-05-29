import tensorflow as tf
from global_vars import*
from global_vars import*
from audio_buffer import*
from compressor import Compressor
from features import*
from cost_function import*
from utility import time_str
import matplotlib.pyplot as plt
from timer import runTimer
from utils import directory_init, idx_to_time, cost_plot, attack_release_plot
from sklearn.preprocessing import StandardScaler
import pickle
from pg_model import predict_times
from audio import saveAudio
from hist_cost import get_histogram_cost

'''Main functions for Policy Gradient implementation'''

def expected_returns(rewards, gamma, standardize=True):
    
    '''rewards -> list of rewards received at each timestep\n
    gamma -> discount rate in [0, 1]\n
    standardize -> shift and scale so rewards are zero mean and unit variance'''
    
    T = tf.shape(rewards)[0]
    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    returns = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    discounted_sum = tf.constant(0.0, dtype='float32')
    discounted_sum_shape = discounted_sum.shape

    for t in tf.range(T):
        reward = rewards[t]
        discounted_sum = reward + gamma * discounted_sum
        discounted_sum.set_shape(discounted_sum_shape)
        returns = returns.write(t, discounted_sum)

    returns = returns.stack()[::-1]

    if standardize:
        mean = tf.math.reduce_mean(returns)
        std = tf.math.reduce_std(returns)
        returns = (returns - mean)/(std + eps)

    return returns

def get_actor_loss(values, returns, attack_probs, release_probs):
    
    '''Computes actor loss using the Policy Gradient Algorithm.
    \nvalues -> predicted V(s) from critic for each time step
    \nreturns -> discounted returns at each time step
    \nattack_probs -> probability for selected attack at each time step
    \nrelease_probs -> probability for selected release at each time step'''

    log_attack_probs = tf.math.log(attack_probs)
    log_release_probs = tf.math.log(release_probs)
    advantage = returns - values
    loss = -tf.math.reduce_sum( (log_attack_probs + log_release_probs) * advantage)
    return loss

def get_critic_loss(values, returns):
    
    '''Computes citic loss using the Policy Gradient Algorithm.
    \nvalues -> predicted V(s) from critic for each time step
    \nreturns -> discounted returns at each time step'''
    
    loss_fcn = tf.keras.losses.MSE
    #loss_fcn = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
    return loss_fcn(values, returns)

def get_loss(values, returns, attack_probs, release_probs):
    actor_loss = get_actor_loss(values, returns, attack_probs, release_probs)
    critic_loss = get_critic_loss(returns, values)
    return tf.cast(actor_loss, 'float32'), tf.cast(critic_loss, 'float32')

def run_episode(model, input_file_path, write_path, time_plot_path, cost_path, gr_path, epoch_number=0):
    
    '''Runs training episode for a given input audio file.
    \nmodel -> PGModel instance to make predictions
    \ninput_file_path -> path to audio file
    \nepoch_number -> current epoch number'''

    if compState: #Select scalar
        scalar_path = 'scalar_pickle_comp'
    else:
        scalar_path = 'scalar_pickle'
    
    audio = AudioBuffer(input_file_path, frame_len, lookahead_frames) #Buffer to read/write audio
    comp = Compressor(thr, ratio) #Compressor
    feature = EnvelopeFinder(history_frames, lookahead_frames, 
    frame_len, history_neurons, lookahead_neurons, Fs, scalar_path) #Envelope finder to extract features

    #TensorArrays to hold values for loss function computation
    attack_probs = tf.TensorArray(dtype='float32', size=0, dynamic_size=True) #Probability of selected attack
    release_probs = tf.TensorArray(dtype='float32', size=0, dynamic_size=True) #Probability of selected release
    values = tf.TensorArray(dtype='float32', size=0, dynamic_size=True) #Critic prediction of state value
    accuracy_costs = tf.TensorArray(dtype='float32', size=0, dynamic_size=True) #Accuracy cost values 
    histogram_costs = tf.TensorArray(dtype='float32', size=0, dynamic_size=True) #Histogram (i.e. rate) cost values
    actives = tf.TensorArray(dtype='bool', size=0, dynamic_size=True) #Only include frames where compressor is active

    weighting_audio = [] #For rate-cost computation
    attack_times = [] #For plotting
    release_times = [] #For plotting
    
    done = False
    sufficient_frames = False #Is envelope finder's buffer full?
    t = 0

    # MAIN EPISODE LOOP #
    while not done: 

        input_frame, lookahead_frame, done = audio.next_frame() #Get next frames
        output_frame, accuracy_cost, active = comp.process_frame(input_frame, sufficient_frames) #Pass through compressor
        audio.write_frame(output_frame) #Write to output buffer
        state, sufficient_frames = feature.update(output_frame, lookahead_frame) #Get state via Hilbert Transform

        if compState:
            state = np.concatenate((comp.get_state(), state))

        if sufficient_frames and not done: #Only predict once sufficient frames have been read
            
            attack, release, attack_prob, release_prob, value = model(state) #Predict using model
            comp.set_times_tf(attack, release) #Set compressor
            
            #Buffer Updates
            weighting_audio.append(input_frame) 
            accuracy_costs = accuracy_costs.write(t, accuracy_cost)
            histogram_costs = histogram_costs.write(t, get_histogram_cost(input_frame, output_frame) )
            actives = actives.write(t, bool(active))
            values = values.write(t, tf.squeeze(value)) 
            attack_probs = attack_probs.write(t, attack_prob)
            release_probs = release_probs.write(t, release_prob)

            #Save time values for plotting
            attack_times.append(attack)
            release_times.append(release)
            
            t += 1
    
    # END OF MAIN EPISODE LOOP #

    attack_release_plot(attack_times, release_times, time_plot_path + "/epoch_{}".format(epoch_number+1)) #Plot attack and release times

    #Compute raw reward values
    #cost_audio = comp.get_costs_and_reset()
    #weighting_audio = np.concatenate( weighting_audio, axis=0 )
    #rate_costs = get_rate_cost_tf(cost_audio, weighting_audio, frame_len)
    #rewards = get_total_reward(rate_costs, accuracy_costs.stack(), cost_weights) #Distortion cost
    rewards = get_total_reward_tf(histogram_costs.stack(), accuracy_costs.stack(), cost_weights) #Histogram cost

    #Generate plots
    cost_plot(histogram_costs.stack(), accuracy_costs.stack(), cost_path + "/epoch_{}".format(epoch_number+1))
    comp.plot_gain_history(gr_path + "/epoch_{}".format(epoch_number))

    #Write output audio to file
    clipped = audio.write_output_to_file(write_path + "/epoch_{}.wav".format(epoch_number+1))
    if clipped: raise RuntimeError("output audio clipped during write")

    actives = actives.stack()
    return (tf.boolean_mask(rewards, actives), tf.boolean_mask(values.stack(), actives), tf.boolean_mask(attack_probs.stack(), actives), tf.boolean_mask(release_probs.stack(), actives))

def train_step(model, input_file_path, write_path, time_plot_path, cost_path, gr_path, lr, epoch_number=0):

    '''Completes one training step for Policy Gradient model.
    \nmodel -> model to train
    \ninput_file_path -> path to training file
    \nwrite_path -> path to same audio output of compressor
    \ntime_plot_path -> location to save plot of attack/release times
    \nlr -> learning rate
    \nepoch_number -> number of current epoch, used to label plots'''

    #Run episode and find losses
    with tf.GradientTape() as ActorTape, tf.GradientTape() as CriticTape:

        ActorTape.watch(model.actor.trainable_variables)
        CriticTape.watch(model.critic.trainable_variables)

        rewards, values, attack_probs, release_probs = run_episode(model, input_file_path, write_path, time_plot_path, cost_path, gr_path, epoch_number)
        returns = expected_returns(rewards, gamma)
        actor_loss = get_actor_loss(values, returns, attack_probs, release_probs)
        critic_loss = get_critic_loss(values, returns)

    #Find gradients
    actor_grads = ActorTape.gradient(actor_loss, model.actor.trainable_variables)
    critic_grads = CriticTape.gradient(critic_loss, model.critic.trainable_variables)

    #Weight updates
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    opt.apply_gradients(zip(actor_grads, model.actor.trainable_variables))
    opt.apply_gradients(zip(critic_grads, model.critic.trainable_variables))

    total_reward = tf.math.reduce_sum(tf.abs(rewards)) #Compute total reward
    total_loss = tf.reduce_sum(tf.abs(actor_loss)) + tf.reduce_sum(tf.abs(critic_loss)) #Compute total loss
    
    N = rewards.shape[0]
    return 1000*total_reward/N, total_loss/N

def make_scalar(input_file_path, pickle_path):

    scalar = StandardScaler()
    audio = AudioBuffer(input_file_path, frame_len, lookahead_frames) #Buffer to read/write audio
    feature = EnvelopeFinder(history_frames, lookahead_frames, frame_len, history_neurons, lookahead_neurons)

    comp = Compressor(thr, ratio) #Compressor
    comp.set_times(10,10) #Initial attack/release

    done = False
    states = []
    while not done:
        input_frame, lookahead_frame, done = audio.next_frame()
        comp.process_frame(input_frame, generate_cost=False)
        state, _ = feature.update(input_frame, lookahead_frame)
        states.append(state)
    
    scalar.fit(states)
    pickle.dump(scalar, open(pickle_path,'wb'))

def run_episode_tf(actor, critic, audio, comp, feature):
    
    '''Runs training episode for a given input audio file.'''

    #TensorArrays to hold values for loss function computation
    attack_probs = tf.TensorArray(dtype='float32', size=0, dynamic_size=True, name='attack_probs') #Probability of selected attack
    release_probs = tf.TensorArray(dtype='float32', size=0, dynamic_size=True, name='relase_probs') #Probability of selected release
    values = tf.TensorArray(dtype='float32', size=0, dynamic_size=True, name='values') #Critic prediction of state value
    accuracy_costs = tf.TensorArray(dtype='float32', size=0, dynamic_size=True, name='accuracy_costs') #Accuracy cost values 
    histogram_costs = tf.TensorArray(dtype='float32', size=0, dynamic_size=True, name='histogram_costs') #Histogram (i.e. rate) cost values
    actives = tf.TensorArray(dtype='int16', size=0, dynamic_size=True, name='actives') #Only include frames where compressor is active

    #TensorArrays for plotting
    attack_times = tf.TensorArray(dtype='float32', size=0, dynamic_size=True, name='attack_times')
    release_times = tf.TensorArray(dtype='float32', size=0, dynamic_size=True, name='release_times')

    done = tf.constant(0, dtype='int16')
    done_shape = done.shape
    
    state = tf.constant(np.zeros(state_len), dtype='float32')
    state_shape = state.shape
    
    active = tf.constant(0, dtype='int16')
    active_shape = active.shape
    
    accuracy_cost = tf.constant(0.0, dtype='float32')
    accuracy_cost_shape = accuracy_cost.shape

    t = tf.constant(0) #Counter for buffer writes

    # MAIN EPISODE LOOP #
    while done == 0: 

        input_frame, lookahead_frame, done = audio.next_frame_tf() #Get next frames
        done.set_shape(done_shape) #Correct shape
        lookahead_state, _ = feature.update_tf(lookahead_frame) #Get state via Hilbert Transform
        comp_state = tf.squeeze(comp.get_state_tf())

        attack, release, attack_prob, release_prob, value = predict_times(comp_state, lookahead_state, actor, critic) #Predict using model
        comp.set_times_tf(attack, release) #Set compressor
        
        output_frame, accuracy_cost, active = comp.process_frame_tf(input_frame, False) #Pass through compressor
        accuracy_cost.set_shape(accuracy_cost_shape) #Correct shape
        active.set_shape(active_shape) #Correct shape
                    
        #Buffer Updates
        accuracy_costs = accuracy_costs.write(t, accuracy_cost)
        histogram_costs = histogram_costs.write(t, get_histogram_cost(input_frame, output_frame) )
        actives = actives.write(t, active)
        values = values.write(t, tf.squeeze(value)) 
        attack_probs = attack_probs.write(t, attack_prob)
        release_probs = release_probs.write(t, release_prob)
        attack_times = attack_times.write(t, attack)
        release_times = release_times.write(t, release)
            
        t = t+1
    
    # END OF MAIN EPISODE LOOP #    

    #Compute raw reward values
    rewards =  get_total_reward(histogram_costs.stack(), accuracy_costs.stack(), cost_weights) #Histogram cost

    actives = actives.stack()
    rewards = tf.squeeze(rewards)
    values = tf.squeeze(values.stack())
    attack_probs = tf.squeeze(attack_probs.stack())
    release_probs = tf.squeeze(release_probs.stack())

    return tf.boolean_mask(rewards, actives), tf.boolean_mask(values, actives), tf.boolean_mask(attack_probs, actives), tf.boolean_mask(release_probs, actives), [histogram_costs.stack(), accuracy_costs.stack(), attack_times.stack(), release_times.stack()]

@tf.function
def train_step_tf(actor, critic, audio, comp, feature, opt, gamma):

    '''Completes one training step for Policy Gradient model.
    \nmodel -> model to train
    \nwrite_path -> path to same audio output of compressor
    \ntime_plot_path -> location to save plot of attack/release times
    \nlr -> learning rate
    \nepoch_number -> number of current epoch, used to label plots'''

    #Run episode and find losses
    with tf.GradientTape() as ActorTape, tf.GradientTape() as CriticTape:

        ActorTape.watch(actor.trainable_variables)
        CriticTape.watch(critic.trainable_variables)
        
        rewards, values, attack_probs, release_probs, plot_data = run_episode_tf(actor, critic, audio, comp, feature)
        returns = expected_returns(rewards, gamma)
        actor_loss, critic_loss = get_loss(values, returns, attack_probs, release_probs)
 
    #Find gradients
    actor_grads = ActorTape.gradient(actor_loss, actor.trainable_variables)
    critic_grads = CriticTape.gradient(critic_loss, critic.trainable_variables)

    #Weight updates
    opt.apply_gradients(zip(actor_grads, actor.trainable_variables))
    opt.apply_gradients(zip(critic_grads, critic.trainable_variables))
    
    total_reward = tf.math.reduce_sum(tf.abs(rewards)) #Compute total reward
    total_loss = tf.reduce_sum(tf.abs(actor_loss)) + tf.reduce_sum(tf.abs(critic_loss)) #Compute total loss
    
    return tf.cast(total_reward, 'float32'), tf.cast(total_loss, 'float32'), plot_data

if __name__ == '__main__':

    make_scalar("../Training Sets/Test Files/Drums 1.wav", "scalar_pickle_comp")
        

