"""
DDPG Trainer for IRS Phase Shift Optimization

This script implements a Deep Deterministic Policy Gradient (DDPG) algorithm for optimizing
IRS phase shifts to:
1. Maximize main lobe and steer it to the user's location
2. Minimize side lobes to prevent energy wastage
3. Optimize user SINR
4. Use IRS path only when SINR is higher than direct path

References:
- DDPG algorithm: Lin et al., 2020, "Deep Reinforcement Learning for Robust Beamforming in IRS-assisted Wireless Communications"
- Channel model: Wu & Zhang, 2019, "Intelligent Reflecting Surface Enhanced Wireless Network via Joint Active and Passive Beamforming"
- IRS beamforming: Wu et al., 2021, "Intelligent Reflecting Surface-Aided Wireless Communications: A Tutorial"
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import json
import os
import time
from tqdm import tqdm
import random
from collections import deque
import io
import sys

# Fix for Unicode encoding errors on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Constants based on IRS literature (Wu & Zhang, 2019; Lin et al., 2020)
CARRIER_FREQUENCY = 2.4e9  # 2.4 GHz
WAVELENGTH = 3e8 / CARRIER_FREQUENCY
NOISE_POWER_DBM = -90  # dBm
TX_POWER_DBM = 20  # dBm
TX_POWER_W = 10**(TX_POWER_DBM/10) / 1000  # Convert dBm to Watts
NUM_IRS_ELEMENTS = 8

# DDPG hyperparameters (based on similar settings in Lin et al., 2020)
ACTOR_LR = 0.0001
CRITIC_LR = 0.001
BATCH_SIZE = 64
GAMMA = 0.99  # Discount factor
TAU = 0.001  # Target network soft update parameter
BUFFER_SIZE = 100000
EXPLORATION_NOISE = 0.1  # Standard deviation of exploration noise
TRAINING_EPISODES = 100  # Train for 100 episodes as specified

# Added parameter to avoid TF warnings about complex data types
TF_COMPLEX_WARNING_FIX = True

class ReplayBuffer:
    """
    Replay buffer for storing and sampling experience tuples.
    Based on standard DRL implementations and used in Lin et al., 2020.
    """
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch_size = min(len(self.buffer), batch_size)
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def size(self):
        return len(self.buffer)

class OUActionNoise:
    """
    Ornstein-Uhlenbeck process noise generator for action exploration.
    Commonly used in DDPG implementations for continuous control tasks.
    """
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()
    
    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x
    
    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

class IRSEnvironment:
    """
    Environment class that simulates the IRS-assisted communication system.
    Adapted from models described in Wu & Zhang, 2019 and Lin et al., 2020.
    """
    def __init__(self, dataset_path="data/irs_dataset.csv", env_path="data/environment.json"):
        # Create placeholder data if files don't exist (for testing purposes)
        self._create_placeholder_data_if_needed(dataset_path, env_path)
        
        try:
            # Load dataset and environment
            self.df = pd.read_csv(dataset_path)
            
            with open(env_path, 'r') as f:
                env_data = json.load(f)
            
            self.access_point = np.array(env_data["access_point"])
            self.irs_elements = np.array(env_data["irs_elements"])
            self.obstacle = {
                "center": np.array(env_data["obstacle"]["center"]),
                "dimensions": np.array(env_data["obstacle"]["dimensions"])
            }
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Using default environment settings instead")
            self._setup_default_environment()
            
        # Environment parameters
        self.num_irs_elements = NUM_IRS_ELEMENTS
        
        # Calculate the number of episodes and frames from the dataset
        if hasattr(self, 'df'):
            self.num_episodes = len(self.df['episode_id'].unique())
            self.frames_per_episode = len(self.df[self.df['episode_id'] == 0])
        else:
            self.num_episodes = 10  # Default number of episodes
            self.frames_per_episode = 50  # Default frames per episode
        
        # Current state tracking
        self.current_episode = 0
        self.current_frame = 0
        self.episode_data = None
        self.frame_data = None
        
        # Reset to initialize environment
        self.reset()
    
    def _setup_default_environment(self):
        """Set up a default environment if loading fails"""
        # Default environment parameters
        self.access_point = np.array([0.0, 0.0, 3.0])  # AP at origin, 3m height
        self.irs_elements = np.array([
            [5.0, 0.0, 2.0 + 0.1*i] for i in range(NUM_IRS_ELEMENTS)
        ])  # IRS elements in a line along z-axis
        self.obstacle = {
            "center": np.array([2.5, 0.0, 1.5]),
            "dimensions": np.array([1.0, 2.0, 3.0])
        }
        
        # Create synthetic dataset
        episodes = []
        for episode_id in range(10):  # 10 episodes
            for frame_id in range(50):  # 50 frames per episode
                # Random user position (varies with time)
                user_pos = np.array([
                    7.0 + 0.5*np.sin(frame_id/10),  # x coordinate
                    3.0 * np.cos(episode_id/5) * np.sin(frame_id/15),  # y coordinate
                    1.5  # z coordinate (constant height)
                ])
                
                # Synthetic channel data
                h_direct_real = 0.1 * np.cos(episode_id + frame_id/10)
                h_direct_imag = 0.1 * np.sin(episode_id + frame_id/10)
                
                h_tx_irs_real = [0.2 * np.cos(episode_id/3 + i/5 + frame_id/20) for i in range(NUM_IRS_ELEMENTS)]
                h_tx_irs_imag = [0.2 * np.sin(episode_id/3 + i/5 + frame_id/20) for i in range(NUM_IRS_ELEMENTS)]
                
                h_irs_rx_real = [0.3 * np.cos(episode_id/4 + i/6 + frame_id/15) for i in range(NUM_IRS_ELEMENTS)]
                h_irs_rx_imag = [0.3 * np.sin(episode_id/4 + i/6 + frame_id/15) for i in range(NUM_IRS_ELEMENTS)]
                
                episodes.append({
                    'episode_id': episode_id,
                    'frame_id': frame_id,
                    'user_position': list(user_pos),
                    'H_direct_real': h_direct_real,
                    'H_direct_imag': h_direct_imag,
                    'H_tx_irs_real': str(h_tx_irs_real),
                    'H_tx_irs_imag': str(h_tx_irs_imag),
                    'H_irs_rx_real': str(h_irs_rx_real),
                    'H_irs_rx_imag': str(h_irs_rx_imag)
                })
        
        # Convert to DataFrame
        self.df = pd.DataFrame(episodes)
    
    def _create_placeholder_data_if_needed(self, dataset_path, env_path):
        """Create placeholder data files if they don't exist"""
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        os.makedirs(os.path.dirname(env_path), exist_ok=True)
        
        # Check if dataset file exists
        if not os.path.exists(dataset_path):
            print(f"Dataset file {dataset_path} not found. Creating a placeholder dataset.")
            # Create a placeholder dataset
            episodes = []
            for episode_id in range(10):  # 10 episodes
                for frame_id in range(50):  # 50 frames per episode
                    # Random user position (varies with time)
                    user_pos = np.array([
                        7.0 + 0.5*np.sin(frame_id/10),  # x coordinate
                        3.0 * np.cos(episode_id/5) * np.sin(frame_id/15),  # y coordinate
                        1.5  # z coordinate (constant height)
                    ])
                    
                    # Synthetic channel data
                    h_direct_real = 0.1 * np.cos(episode_id + frame_id/10)
                    h_direct_imag = 0.1 * np.sin(episode_id + frame_id/10)
                    
                    h_tx_irs_real = [0.2 * np.cos(episode_id/3 + i/5 + frame_id/20) for i in range(NUM_IRS_ELEMENTS)]
                    h_tx_irs_imag = [0.2 * np.sin(episode_id/3 + i/5 + frame_id/20) for i in range(NUM_IRS_ELEMENTS)]
                    
                    h_irs_rx_real = [0.3 * np.cos(episode_id/4 + i/6 + frame_id/15) for i in range(NUM_IRS_ELEMENTS)]
                    h_irs_rx_imag = [0.3 * np.sin(episode_id/4 + i/6 + frame_id/15) for i in range(NUM_IRS_ELEMENTS)]
                    
                    episodes.append({
                        'episode_id': episode_id,
                        'frame_id': frame_id,
                        'user_position': str(list(user_pos)),
                        'H_direct_real': h_direct_real,
                        'H_direct_imag': h_direct_imag,
                        'H_tx_irs_real': str(h_tx_irs_real),
                        'H_tx_irs_imag': str(h_tx_irs_imag),
                        'H_irs_rx_real': str(h_irs_rx_real),
                        'H_irs_rx_imag': str(h_irs_rx_imag)
                    })
            
            # Create DataFrame and save to CSV
            df = pd.DataFrame(episodes)
            df.to_csv(dataset_path, index=False)
        
        # Check if environment file exists
        if not os.path.exists(env_path):
            print(f"Environment file {env_path} not found. Creating a placeholder environment.")
            # Create a placeholder environment
            env_data = {
                "access_point": [0.0, 0.0, 3.0],  # AP at origin, 3m height
                "irs_elements": [[5.0, 0.0, 2.0 + 0.1*i] for i in range(NUM_IRS_ELEMENTS)],  # IRS elements in a line along z-axis
                "obstacle": {
                    "center": [2.5, 0.0, 1.5],
                    "dimensions": [1.0, 2.0, 3.0]
                }
            }
            
            # Save to JSON
            with open(env_path, 'w') as f:
                json.dump(env_data, f, indent=4)
    
    def reset(self, episode_id=None):
        """Reset environment to start of an episode"""
        if episode_id is None:
            # Randomly select an episode
            episode_id = np.random.randint(0, self.num_episodes)
        
        self.current_episode = episode_id
        self.current_frame = 0
        
        # Get data for this episode
        self.episode_data = self.df[self.df['episode_id'] == episode_id].sort_values('frame_id')
        
        # Get first frame
        self.frame_data = self.episode_data.iloc[0]
        
        # Prepare state (user position, channel information)
        state = self._get_state()
        
        return state
    
    def _get_state(self):
        """Extract state from current frame data"""
        try:
            # Extract user position (safely)
            user_pos_str = self.frame_data['user_position']
            try:
                user_pos = np.array(eval(user_pos_str))
            except:
                # If eval fails, try to parse as a list
                import re
                user_pos_str = re.sub(r'[\[\]]', '', user_pos_str)
                user_pos = np.array([float(x) for x in user_pos_str.split(',')])
        except Exception as e:
            print(f"Error parsing user position: {e}")
            user_pos = np.array([0.0, 0.0, 0.0])
        
        # Extract channel information (safely)
        try:
            h_direct_real = float(self.frame_data['H_direct_real'])
            h_direct_imag = float(self.frame_data['H_direct_imag'])
        except:
            h_direct_real = 0.0
            h_direct_imag = 0.0
        
        try:
            h_tx_irs_real_str = self.frame_data['H_tx_irs_real']
            h_tx_irs_imag_str = self.frame_data['H_tx_irs_imag']
            
            try:
                h_tx_irs_real = np.array(eval(h_tx_irs_real_str))
                h_tx_irs_imag = np.array(eval(h_tx_irs_imag_str))
            except:
                # If eval fails, try to parse as a list
                import re
                h_tx_irs_real_str = re.sub(r'[\[\]]', '', h_tx_irs_real_str)
                h_tx_irs_imag_str = re.sub(r'[\[\]]', '', h_tx_irs_imag_str)
                
                h_tx_irs_real = np.array([float(x) for x in h_tx_irs_real_str.split(',')])
                h_tx_irs_imag = np.array([float(x) for x in h_tx_irs_imag_str.split(',')])
        except Exception as e:
            print(f"Error parsing H_tx_irs: {e}")
            h_tx_irs_real = np.zeros(self.num_irs_elements)
            h_tx_irs_imag = np.zeros(self.num_irs_elements)
        
        try:
            h_irs_rx_real_str = self.frame_data['H_irs_rx_real']
            h_irs_rx_imag_str = self.frame_data['H_irs_rx_imag']
            
            try:
                h_irs_rx_real = np.array(eval(h_irs_rx_real_str))
                h_irs_rx_imag = np.array(eval(h_irs_rx_imag_str))
            except:
                # If eval fails, try to parse as a list
                import re
                h_irs_rx_real_str = re.sub(r'[\[\]]', '', h_irs_rx_real_str)
                h_irs_rx_imag_str = re.sub(r'[\[\]]', '', h_irs_rx_imag_str)
                
                h_irs_rx_real = np.array([float(x) for x in h_irs_rx_real_str.split(',')])
                h_irs_rx_imag = np.array([float(x) for x in h_irs_rx_imag_str.split(',')])
        except Exception as e:
            print(f"Error parsing H_irs_rx: {e}")
            h_irs_rx_real = np.zeros(self.num_irs_elements)
            h_irs_rx_imag = np.zeros(self.num_irs_elements)
        
        # Combine into state vector
        state = np.concatenate([
            user_pos,  # User position
            [h_direct_real, h_direct_imag],  # Direct channel
            h_tx_irs_real, h_tx_irs_imag,  # Tx-IRS channels
            h_irs_rx_real, h_irs_rx_imag  # IRS-Rx channels
        ])
        
        return state
    
    def step(self, action):
        """Take action (IRS phase shifts) and return next state, reward, done"""
        # Apply phase shifts (action) to calculate effective channel
        h_direct_complex = complex(self.frame_data['H_direct_real'], self.frame_data['H_direct_imag'])
        
        try:
            h_tx_irs_real = np.array(eval(self.frame_data['H_tx_irs_real']))
            h_tx_irs_imag = np.array(eval(self.frame_data['H_tx_irs_imag']))
            h_tx_irs_complex = np.array([complex(r, i) for r, i in zip(h_tx_irs_real, h_tx_irs_imag)])
            
            h_irs_rx_real = np.array(eval(self.frame_data['H_irs_rx_real']))
            h_irs_rx_imag = np.array(eval(self.frame_data['H_irs_rx_imag']))
            h_irs_rx_complex = np.array([complex(r, i) for r, i in zip(h_irs_rx_real, h_irs_rx_imag)])
        except Exception as e:
            print(f"Error parsing channel data: {e}")
            # Fallback to default values
            h_tx_irs_complex = np.array([complex(0.2, 0.2) for _ in range(self.num_irs_elements)])
            h_irs_rx_complex = np.array([complex(0.3, 0.3) for _ in range(self.num_irs_elements)])
        
        # Calculate effective channel with given phase shifts
        phase_shifts_complex = np.exp(1j * action)
        irs_component = np.sum(h_tx_irs_complex * phase_shifts_complex * h_irs_rx_complex)
        effective_channel = h_direct_complex + irs_component
        
        # Calculate SNR with IRS
        noise_power_w = 10**(NOISE_POWER_DBM/10) / 1000
        received_power = TX_POWER_W * (abs(effective_channel) ** 2)
        snr = received_power / noise_power_w
        snr_db = 10 * np.log10(snr)
        
        # Calculate SNR for direct path only
        direct_power = TX_POWER_W * (abs(h_direct_complex) ** 2)
        direct_snr = direct_power / noise_power_w
        direct_snr_db = 10 * np.log10(direct_snr)
        
        # Calculate radiation pattern and main lobe direction
        # This is a simplified model - in practice, this would be more complex
        try:
            user_pos = np.array(eval(self.frame_data['user_position']))
        except:
            # Fallback if parsing fails
            user_pos = np.array([7.0, 0.0, 1.5])
            
        user_angle = np.arctan2(user_pos[1], user_pos[0])  # Angle from AP to user
        
        # Calculate beam pattern with given phase shifts
        angles = np.linspace(-np.pi, np.pi, 360)
        pattern = np.zeros_like(angles)
        
        # Calculate radiation pattern (simplified model)
        for i, angle in enumerate(angles):
            # Array factor calculation (simplified)
            array_factor = 0
            for j in range(self.num_irs_elements):
                # Element position relative to IRS center
                element_offset = self.irs_elements[j] - np.mean(self.irs_elements, axis=0)
                # Phase contribution at this angle
                phase = action[j] + (2 * np.pi / WAVELENGTH) * element_offset[2] * np.sin(angle)
                array_factor += np.exp(1j * phase)
            
            pattern[i] = abs(array_factor)
        
        # Normalize pattern
        pattern = pattern / np.max(pattern) if np.max(pattern) > 0 else pattern
        
        # Find main lobe angle (angle of maximum radiation)
        main_lobe_idx = np.argmax(pattern)
        main_lobe_angle = angles[main_lobe_idx]
        
        # Find side lobes (local maxima)
        side_lobe_mask = np.r_[False, pattern[1:-1] > pattern[:-2]] & np.r_[pattern[1:-1] > pattern[2:], False]
        side_lobe_indices = np.where(side_lobe_mask)[0]
        side_lobe_values = pattern[side_lobe_indices]
        
        # Calculate maximum side lobe level
        if len(side_lobe_values) > 0:
            max_side_lobe = np.max(side_lobe_values)
        else:
            max_side_lobe = 0
            
        # Reward calculation has three components:
        # 1. SNR improvement compared to direct path
        snr_improvement = max(0, snr_db - direct_snr_db)  # Only reward if IRS improves SNR
        
        # 2. Main lobe pointing accuracy (angular error)
        angle_error = abs(main_lobe_angle - user_angle)
        if angle_error > np.pi:
            angle_error = 2 * np.pi - angle_error
        angle_accuracy = 1.0 - (angle_error / np.pi)  # 1 when perfect, 0 when worst
        
        # 3. Side lobe suppression
        side_lobe_suppression = 1.0 - max_side_lobe  # 1 when no side lobes, 0 when side lobe same as main lobe
        
        # Combined reward
        reward = (0.5 * snr_improvement) + (0.3 * angle_accuracy) + (0.2 * side_lobe_suppression)
        
        # Move to next frame
        self.current_frame += 1
        done = self.current_frame >= len(self.episode_data)
        
        # Get next state if not done
        if not done:
            self.frame_data = self.episode_data.iloc[self.current_frame]
            next_state = self._get_state()
        else:
            next_state = self._get_state()  # Just return current state if done
        
        # Additional info for debugging and visualization
        info = {
            'snr_db': snr_db,
            'direct_snr_db': direct_snr_db,
            'effective_channel': effective_channel,
            'main_lobe_angle': main_lobe_angle,
            'user_angle': user_angle,
            'pattern': pattern,
            'angles': angles,
            'max_side_lobe': max_side_lobe,
            'snr_improvement': snr_improvement,
            'angle_accuracy': angle_accuracy,
            'side_lobe_suppression': side_lobe_suppression
        }
        
        return next_state, reward, done, info

class DDPG:
    """
    Deep Deterministic Policy Gradient implementation for IRS phase shift optimization.
    Based on architecture described in Lin et al., 2020.
    """
    def __init__(self, state_dim, action_dim, action_bound_high):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound_high = action_bound_high  # Upper bound for actions (2π for phase shifts)
        
        # Initialize actor and critic networks (both main and target)
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        self.target_actor = self._build_actor()
        self.target_critic = self._build_critic()
        
        # Copy weights to target networks
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())
        
        # Optimizers
        self.actor_optimizer = Adam(learning_rate=ACTOR_LR)
        self.critic_optimizer = Adam(learning_rate=CRITIC_LR)
        
        # Noise process for action exploration
        self.noise = OUActionNoise(
            mean=np.zeros(action_dim),
            std_deviation=EXPLORATION_NOISE * np.ones(action_dim)
        )
        
        # Replay buffer
        self.buffer = ReplayBuffer(BUFFER_SIZE)
        
        # Training metrics
        self.actor_loss_history = []
        self.critic_loss_history = []
        self.reward_history = []
        self.snr_history = []
        
        # Flag to check if buffer has enough samples
        self.buffer_ready = False
    
    def _build_actor(self):
        """Build actor network that predicts actions given states"""
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        
        inputs = Input(shape=(self.state_dim,))
        x = Dense(256, activation='relu')(inputs)
        x = Dense(128, activation='relu')(x)
        outputs = Dense(self.action_dim, activation='tanh', kernel_initializer=last_init)(x)
        
        # Scale outputs to [0, 2π] range for phase shifts
        outputs = outputs * self.action_bound_high
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def _build_critic(self):
        """Build critic network that predicts Q-values given state-action pairs"""
        # State input
        state_input = Input(shape=(self.state_dim,))
        state_out = Dense(128, activation='relu')(state_input)
        
        # Action input
        action_input = Input(shape=(self.action_dim,))
        action_out = Dense(128, activation='relu')(action_input)
        
        # Concatenate state and action pathways
        concat = Concatenate()([state_out, action_out])
        x = Dense(128, activation='relu')(concat)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(1)(x)  # Q-value output
        
        model = Model(inputs=[state_input, action_input], outputs=outputs)
        return model
    
    def policy(self, state, add_noise=True):
        """Get action from actor network with optional exploration noise"""
        state = np.reshape(state, (1, -1))
        
        # Handle NaN or inf values in state
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            print("Warning: NaN or inf values in state. Replacing with zeros.")
            state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Use TF's eager execution to avoid potential issues with complex numbers
        if TF_COMPLEX_WARNING_FIX:
            tf_state = tf.convert_to_tensor(state, dtype=tf.float32)
            action = self.actor(tf_state).numpy()[0]
        else:
            action = self.actor.predict(state, verbose=0)[0]
        
        if add_noise:
            noise = self.noise()
            action += noise
        
        # Ensure actions are within bounds [0, 2π]
        action = np.clip(action, 0, self.action_bound_high)
        
        return action
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        # Handle NaN or inf values
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
        next_state = np.nan_to_num(next_state, nan=0.0, posinf=0.0, neginf=0.0)
        reward = float(np.nan_to_num(reward, nan=0.0, posinf=10.0, neginf=-10.0))
        
        self.buffer.add(state, action, reward, next_state, done)
        
        # Set flag when buffer has enough samples for training
        if self.buffer.size() >= BATCH_SIZE and not self.buffer_ready:
            self.buffer_ready = True
            print(f"Replay buffer ready with {self.buffer.size()} samples")

    def train(self):
        """Train actor and critic networks using sampled batch"""
        if self.buffer.size() < BATCH_SIZE:
            return
            
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)
        
        # Handle NaN values in batch
        states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)
        actions = np.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
        rewards = np.nan_to_num(rewards, nan=0.0, posinf=10.0, neginf=-10.0)
        next_states = np.nan_to_num(next_states, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Convert to TensorFlow tensors
        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
        actions_tensor = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards_tensor = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states_tensor = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones_tensor = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        # Train critic
        with tf.GradientTape() as tape:
            # Get target actions from target actor
            target_actions = self.target_actor(next_states_tensor)
            
            # Get target Q-values from target critic
            target_q_values = self.target_critic([next_states_tensor, target_actions])
            
            # Calculate target using Bellman equation
            targets = rewards_tensor + (1 - dones_tensor) * GAMMA * target_q_values
            
            # Get current Q-values predictions
            current_q_values = self.critic([states_tensor, actions_tensor])
            
            # Calculate critic loss (MSE)
            critic_loss = tf.reduce_mean(tf.square(targets - current_q_values))
        
        # Get critic gradients and update weights
        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))
        
        # Train actor
        with tf.GradientTape() as tape:
            # Get actions from actor
            actor_actions = self.actor(states_tensor)
            
            # Get Q-values from critic
            actor_q_values = self.critic([states_tensor, actor_actions])
            
            # Calculate actor loss (negative of Q-values)
            actor_loss = -tf.reduce_mean(actor_q_values)
        
        # Get actor gradients and update weights
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
        
        # Update target networks with soft update
        self._update_target_networks()
        
        # Store losses for tracking
        self.actor_loss_history.append(float(actor_loss))
        self.critic_loss_history.append(float(critic_loss))
        
        return float(actor_loss), float(critic_loss)
    
    def _update_target_networks(self):
        """Update target networks using soft update rule"""
        # Update target actor
        for source_var, target_var in zip(self.actor.variables, self.target_actor.variables):
            target_var.assign((1 - TAU) * target_var + TAU * source_var)
        
        # Update target critic
        for source_var, target_var in zip(self.critic.variables, self.target_critic.variables):
            target_var.assign((1 - TAU) * target_var + TAU * source_var)
    
    def save_models(self, path="models"):
        """Save actor and critic models"""
        if not os.path.exists(path):
            os.makedirs(path)
        
        self.actor.save(f"{path}/ddpg_actor.h5")
        self.critic.save(f"{path}/ddpg_critic.h5")
        print(f"Models saved to {path}")
    
    def load_models(self, path="models"):
        """Load actor and critic models"""
        try:
            self.actor = tf.keras.models.load_model(f"{path}/ddpg_actor.h5")
            self.critic = tf.keras.models.load_model(f"{path}/ddpg_critic.h5")
            self.target_actor.set_weights(self.actor.get_weights())
            self.target_critic.set_weights(self.critic.get_weights())
            print(f"Models loaded from {path}")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False


def train_ddpg(episodes=TRAINING_EPISODES, max_steps=None, visualize=False):
    """Train DDPG agent for IRS phase shift optimization"""
    print("Setting up environment and DDPG agent...")
    
    # Create environment
    env = IRSEnvironment()
    
    # Get state and action dimensions
    state = env.reset()
    state_dim = len(state)
    action_dim = env.num_irs_elements
    
    # Action bounds: Phase shifts between 0 and 2π
    action_bound_high = 2 * np.pi
    
    # Create DDPG agent
    agent = DDPG(state_dim, action_dim, action_bound_high)
    
    # Training metrics
    total_steps = 0
    episode_rewards = []
    avg_snr_improvements = []
    
    print(f"Starting training for {episodes} episodes...")
    
    # Progress bar for training
    progress_bar = tqdm(range(episodes), desc="Training")
    
    for episode in progress_bar:
        # Reset environment and noise process
        state = env.reset()
        agent.noise.reset()
        
        episode_reward = 0
        episode_snr_improvements = []
        done = False
        step = 0
        
        # Run episode
        while not done:
            # Select action
            action = agent.policy(state)
            
            # Apply action to environment
            next_state, reward, done, info = env.step(action)
            
            # Store experience in replay buffer
            agent.remember(state, action, reward, next_state, done)
            
            # Update state and metrics
            state = next_state
            episode_reward += reward
            episode_snr_improvements.append(info['snr_improvement'])
            
            # Train agent if buffer has enough samples
            if agent.buffer_ready:
                agent.train()
            
            # Visualize if requested (once every few episodes)
            if visualize and episode % 10 == 0 and step == 0:
                visualize_beam_pattern(info['angles'], info['pattern'], info['user_angle'], info['main_lobe_angle'], episode)
            
            # Limit episode length if specified
            step += 1
            total_steps += 1
            if max_steps and step >= max_steps:
                break
        
        # Record episode metrics
        episode_rewards.append(episode_reward)
        avg_snr_improvement = np.mean(episode_snr_improvements)
        avg_snr_improvements.append(avg_snr_improvement)
        
        # Update progress bar
        progress_bar.set_postfix({
            'reward': f"{episode_reward:.2f}",
            'avg_snr_imp': f"{avg_snr_improvement:.2f}dB"
        })
        
        # Save model periodically
        if (episode + 1) % 20 == 0:
            agent.save_models()
            
            # Plot training curves
            plot_training_curves(agent.actor_loss_history, agent.critic_loss_history, 
                                episode_rewards, avg_snr_improvements, episode+1)
    
    # Final save
    agent.save_models()
    
    # Final plots
    plot_training_curves(agent.actor_loss_history, agent.critic_loss_history, 
                        episode_rewards, avg_snr_improvements, episodes)
    
    print(f"Training completed after {total_steps} total steps")
    return agent


def visualize_beam_pattern(angles, pattern, user_angle, main_lobe_angle, episode):
    """Visualize beam radiation pattern"""
    plt.figure(figsize=(10, 6))
    
    # Convert to degrees for better readability
    angles_deg = np.degrees(angles)
    user_angle_deg = np.degrees(user_angle)
    main_lobe_angle_deg = np.degrees(main_lobe_angle)
    
    # Plot radiation pattern
    plt.plot(angles_deg, pattern)
    
    # Highlight user direction and main lobe
    plt.axvline(x=user_angle_deg, color='r', linestyle='--', label=f'User Direction ({user_angle_deg:.1f}°)')
    plt.axvline(x=main_lobe_angle_deg, color='g', linestyle='--', label=f'Main Lobe ({main_lobe_angle_deg:.1f}°)')
    
    plt.title(f'IRS Beam Pattern (Episode {episode})')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Normalized Amplitude')
    plt.grid(True)
    plt.legend()
    
    # Save figure
    os.makedirs('figures', exist_ok=True)
    plt.savefig(f'figures/beam_pattern_episode_{episode}.png')
    plt.close()


def plot_training_curves(actor_losses, critic_losses, rewards, snr_improvements, episodes):
    """Plot training metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Actor loss
    axes[0, 0].plot(actor_losses)
    axes[0, 0].set_title('Actor Loss')
    axes[0, 0].set_xlabel('Training Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)
    
    # Critic loss
    axes[0, 1].plot(critic_losses)
    axes[0, 1].set_title('Critic Loss')
    axes[0, 1].set_xlabel('Training Step')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True)
    
    # Episode rewards
    axes[1, 0].plot(rewards)
    axes[1, 0].set_title('Episode Rewards')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Total Reward')
    axes[1, 0].grid(True)
    
    # SNR improvements
    axes[1, 1].plot(snr_improvements)
    axes[1, 1].set_title('Average SNR Improvement')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('SNR Improvement (dB)')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs('figures', exist_ok=True)
    plt.savefig(f'figures/training_curves_episode_{episodes}.png')
    plt.close()


def evaluate_agent(agent, episodes=10, visualize=True):
    """Evaluate trained agent performance"""
    print(f"Evaluating agent for {episodes} episodes...")
    
    # Create environment
    env = IRSEnvironment()
    
    # Metrics
    episode_rewards = []
    snr_improvements = []
    beam_accuracies = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        episode_snr_improvements = []
        episode_beam_accuracies = []
        
        while not done:
            # Select action without exploration noise
            action = agent.policy(state, add_noise=False)
            
            # Apply action
            next_state, reward, done, info = env.step(action)
            
            # Update metrics
            state = next_state
            episode_reward += reward
            episode_snr_improvements.append(info['snr_improvement'])
            episode_beam_accuracies.append(info['angle_accuracy'])
            
            # Visualize (first step of each episode)
            if visualize and len(episode_beam_accuracies) == 1:
                visualize_beam_pattern(info['angles'], info['pattern'], 
                                      info['user_angle'], info['main_lobe_angle'], 
                                      f"eval_{episode}")
        
        # Record episode metrics
        episode_rewards.append(episode_reward)
        snr_improvements.append(np.mean(episode_snr_improvements))
        beam_accuracies.append(np.mean(episode_beam_accuracies))
        
        print(f"Episode {episode}: Reward = {episode_reward:.2f}, "
              f"SNR Improvement = {np.mean(episode_snr_improvements):.2f}dB, "
              f"Beam Accuracy = {np.mean(episode_beam_accuracies)*100:.1f}%")
    
    # Print overall performance
    print("\nEvaluation Results:")
    print(f"Average Episode Reward: {np.mean(episode_rewards):.2f}")
    print(f"Average SNR Improvement: {np.mean(snr_improvements):.2f}dB")
    print(f"Average Beam Pointing Accuracy: {np.mean(beam_accuracies)*100:.1f}%")
    
    return {
        'rewards': episode_rewards,
        'snr_improvements': snr_improvements,
        'beam_accuracies': beam_accuracies
    }


def main():
    """Main function to train and evaluate DDPG agent"""
    # Fix for Unicode encoding errors on Windows
    import locale
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Train DDPG agent for IRS Phase Shift Optimization')
    parser.add_argument('--train', action='store_true', help='Train the agent')
    parser.add_argument('--eval', action='store_true', help='Evaluate the agent')
    parser.add_argument('--episodes', type=int, default=TRAINING_EPISODES, help='Number of episodes')
    parser.add_argument('--visualize', action='store_true', help='Visualize training')
    parser.add_argument('--load', action='store_true', help='Load existing model')
    args = parser.parse_args()
    
    # Set default mode to train if none specified
    if not args.train and not args.eval:
        args.train = True
    
    # Create agent
    env = IRSEnvironment()
    state = env.reset()
    state_dim = len(state)
    action_dim = env.num_irs_elements
    action_bound_high = 2 * np.pi
    
    agent = DDPG(state_dim, action_dim, action_bound_high)
    
    # Load existing model if requested
    if args.load:
        agent.load_models()
    
    # Train agent
    if args.train:
        agent = train_ddpg(episodes=args.episodes, visualize=args.visualize)
    
    # Evaluate agent
    if args.eval:
        evaluate_agent(agent, episodes=10, visualize=True)
    
    return agent


if __name__ == "__main__":
    agent = train_ddpg()
    evaluate_agent(agent)
        