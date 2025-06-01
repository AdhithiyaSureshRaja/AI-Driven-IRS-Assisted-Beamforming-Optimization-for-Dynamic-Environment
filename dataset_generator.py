"""
Dataset Generator for IRS Environment

This script generates a synthetic dataset for IRS-assisted communications with:
- 1000 episodes with 100 frames each
- Each episode has:
  * 1 IRS panel with 8 elements (fixed location)
  * 1 Access Point (fixed location)
  * 1 Rectangular obstacle (fixed location)
  * 1 Moving user with different trajectory per episode

References:
- Channel model: Wu & Zhang, 2019, "Intelligent Reflecting Surface Enhanced Wireless Network via Joint Active and Passive Beamforming"
- IRS Phase shifts: Wu et al., 2021, "Intelligent Reflecting Surface-Aided Wireless Communications: A Tutorial"
- SNR calculations: Lin et al., 2020, "Deep Reinforcement Learning for Robust Beamforming in IRS-assisted Wireless Communications"
"""

import numpy as np
import pandas as pd
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Constants based on common IRS literature (Wu & Zhang, 2019; Lin et al., 2020)
CARRIER_FREQUENCY = 2.4e9  # 2.4 GHz
WAVELENGTH = 3e8 / CARRIER_FREQUENCY
NOISE_POWER_DBM = -90  # dBm
TX_POWER_DBM = 20  # dBm
TX_POWER_W = 10**(TX_POWER_DBM/10) / 1000  # Convert dBm to Watts
NUM_IRS_ELEMENTS = 8
NUM_EPISODES = 1000
FRAMES_PER_EPISODE = 100

# Path loss exponents based on common values in literature
PATH_LOSS_EXPONENT_DIRECT = 3.5  # Higher for direct path (more loss)
PATH_LOSS_EXPONENT_IRS = 2.2  # Lower for IRS path (less loss)

def generate_environment():
    """
    Generate fixed components of the environment (IRS panel, AP, obstacle)
    Based on common IRS simulation setups (Wu & Zhang, 2019)
    """
    # Fixed positions for all episodes - coordinate system in meters
    access_point = np.array([0, 0, 5])  # Access point at origin, 5m height
    irs_panel_center = np.array([50, 0, 10])  # IRS panel 50m away, 10m height
    
    # Generate obstacle - mimicking a building between AP and user paths
    # Based on common scenarios in IRS literature where obstacles create NLOS conditions
    obstacle = {
        'center': np.array([25, 10, 7.5]),  # Between AP and user paths
        'dimensions': np.array([10, 15, 15])  # Width, length, height
    }
    
    # Generate positions for each IRS element
    # Elements arranged in a uniform linear array (ULA) - common in IRS studies
    irs_elements = []
    element_spacing = WAVELENGTH/2  # Half-wavelength spacing (standard)
    
    # Generate 8 IRS elements in a vertical array
    for i in range(NUM_IRS_ELEMENTS):
        position = irs_panel_center + np.array([0, 0, -NUM_IRS_ELEMENTS/2*element_spacing + i*element_spacing])
        irs_elements.append(position)
    
    return access_point, np.array(irs_elements), obstacle

def check_los(start_point, end_point, obstacle):
    """
    Check if line of sight is blocked by the obstacle
    Using ray-box intersection test
    """
    # Ray-box intersection algorithm
    # Based on common computer graphics algorithms for ray tracing
    
    # Get obstacle bounds
    obs_min = obstacle['center'] - obstacle['dimensions']/2
    obs_max = obstacle['center'] + obstacle['dimensions']/2
    
    # Direction vector
    direction = end_point - start_point
    direction_norm = np.linalg.norm(direction)
    direction = direction / direction_norm if direction_norm > 0 else direction
    
    # Avoid division by zero
    inv_dir = np.array([1.0/d if abs(d) > 1e-10 else 1e10 for d in direction])
    
    # Find intersection distances
    t1 = (obs_min[0] - start_point[0]) * inv_dir[0]
    t2 = (obs_max[0] - start_point[0]) * inv_dir[0]
    t3 = (obs_min[1] - start_point[1]) * inv_dir[1]
    t4 = (obs_max[1] - start_point[1]) * inv_dir[1]
    t5 = (obs_min[2] - start_point[2]) * inv_dir[2]
    t6 = (obs_max[2] - start_point[2]) * inv_dir[2]
    
    tmin = max(min(t1, t2), min(t3, t4), min(t5, t6))
    tmax = min(max(t1, t2), max(t3, t4), max(t5, t6))
    
    # If tmax < 0 or tmin > tmax or tmin > ray length, no intersection
    if tmax < 0 or tmin > tmax or tmin > 1:
        return False  # No collision
    
    return True  # Collision detected

def generate_user_trajectory(access_point, irs_panel, obstacle, episode_id):
    """
    Generate user movement trajectory for an episode
    Based on mobility models in IRS literature (Wu et al., 2021)
    """
    # Set a random starting point in the viable area
    # We'll use a semi-circular area opposite the IRS panel
    radius = np.random.uniform(20, 60)
    angle = np.random.uniform(-np.pi/2, np.pi/2)  # Half circle opposite IRS
    
    # Random height between 1-2m (typical human height)
    height = np.random.uniform(1.0, 2.0)
    
    # Starting position
    start_x = access_point[0] + radius * np.cos(angle)
    start_y = access_point[1] + radius * np.sin(angle)
    start_pos = np.array([start_x, start_y, height])
    
    # Generate random velocity components (reasonable walking/slow movement speeds)
    speed = np.random.uniform(0.5, 2.0)  # 0.5-2 m/s (walking speed)
    direction = np.random.uniform(0, 2*np.pi)
    
    vx = speed * np.cos(direction)
    vy = speed * np.sin(direction)
    vz = 0  # Assume user stays at same height
    
    velocity = np.array([vx, vy, vz])
    
    # Generate trajectory
    trajectory = []
    velocities = []
    positions = [start_pos]
    current_pos = start_pos.copy()
    
    # Occasionally change direction with small probability
    for frame in range(FRAMES_PER_EPISODE):
        # Small chance to change direction slightly
        if np.random.random() < 0.1:
            angle_change = np.random.uniform(-np.pi/6, np.pi/6)  # Max 30 degree change
            speed_change = np.random.uniform(0.9, 1.1)  # Max 10% speed change
            
            speed *= speed_change
            direction += angle_change
            
            vx = speed * np.cos(direction)
            vy = speed * np.sin(direction)
            velocity = np.array([vx, vy, vz])
        
        # Update position
        current_pos = current_pos + velocity
        
        # Boundary check - keep user in reasonable area
        dist_from_origin = np.linalg.norm(current_pos[:2])
        if dist_from_origin > 70:
            # If going too far, reverse direction back toward origin
            direction = np.arctan2(-current_pos[1], -current_pos[0]) + np.random.uniform(-np.pi/6, np.pi/6)
            vx = speed * np.cos(direction)
            vy = speed * np.sin(direction)
            velocity = np.array([vx, vy, vz])
            
            # Apply the new velocity
            current_pos = current_pos + velocity
        
        # Keep height constant
        current_pos[2] = height
        
        positions.append(current_pos.copy())
        velocities.append(velocity.copy())
    
    # Remove the extra position we added
    positions = positions[:-1]
    
    return np.array(positions), np.array(velocities)

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return np.linalg.norm(point1 - point2)

def calculate_channel_coefficient(tx_pos, rx_pos, is_irs_path=False):
    """
    Calculate complex channel coefficient based on Rayleigh fading model
    Based on channel models in Wu & Zhang, 2019 and Lin et al., 2020
    """
    # Calculate distance
    distance = calculate_distance(tx_pos, rx_pos)
    
    # Path loss exponent depends on the path type
    path_loss_exponent = PATH_LOSS_EXPONENT_IRS if is_irs_path else PATH_LOSS_EXPONENT_DIRECT
    
    # Calculate path loss (standard path loss formula)
    path_loss = (WAVELENGTH / (4 * np.pi * distance)) ** path_loss_exponent
    
    # Rayleigh fading - complex Gaussian with zero mean and unit variance
    h_real = np.random.normal(0, 1/np.sqrt(2))
    h_imag = np.random.normal(0, 1/np.sqrt(2))
    
    # Combine path loss with fading
    channel_coef = np.sqrt(path_loss) * complex(h_real, h_imag)
    
    return channel_coef

def calculate_effective_channel(h_direct, h_tx_irs, h_irs_rx, phase_shifts):
    """
    Calculate the effective channel with IRS phase shifts
    Based on equation from Wu et al., 2021, "Intelligent Reflecting Surface-Aided Wireless Communications: A Tutorial"
    """
    # Convert phase shifts to complex exponentials
    phase_shifts_complex = np.exp(1j * phase_shifts)
    
    # Calculate IRS-aided channel component (element-wise multiplication and sum)
    irs_component = np.sum(h_tx_irs * phase_shifts_complex * h_irs_rx)
    
    # Total effective channel is direct channel + IRS component
    effective_channel = h_direct + irs_component
    
    return effective_channel

def calculate_snr(channel_coefficient, tx_power_w=TX_POWER_W, noise_power_dbm=NOISE_POWER_DBM):
    """
    Calculate SNR in dB
    Based on SNR calculations in Lin et al., 2020
    """
    # Convert noise power from dBm to watts
    noise_power_w = 10**(noise_power_dbm/10) / 1000
    
    # Calculate received power
    received_power = tx_power_w * (abs(channel_coefficient) ** 2)
    
    # Calculate SNR
    snr = received_power / noise_power_w
    
    # Convert to dB
    snr_db = 10 * np.log10(snr)
    
    # Calculate received power in dBm
    received_power_dbm = 10 * np.log10(received_power * 1000)
    
    return snr_db, received_power_dbm

def generate_random_phase_shifts():
    """Generate random phase shifts for IRS elements"""
    return np.random.uniform(0, 2*np.pi, NUM_IRS_ELEMENTS)

def generate_optimal_phase_shifts(h_tx_irs, h_irs_rx):
    """
    Generate optimal phase shifts to maximize SNR
    Based on the phase alignment principle described in Wu et al., 2021
    """
    # Calculate the phase of the product of the two channels
    product_phase = np.angle(h_tx_irs * np.conj(h_irs_rx))
    
    # The optimal phase shift is the negative of this phase
    # This aligns all the reflected signals constructively at the receiver
    optimal_phase_shifts = -product_phase
    
    return optimal_phase_shifts

def generate_dataset():
    """Main function to generate the dataset"""
    print("Generating IRS communication dataset...")
    
    # Create directory for the dataset if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Generate fixed environment components
    access_point, irs_elements, obstacle = generate_environment()
    
    # Save environment setup for reference
    env_data = {
        "access_point": access_point.tolist(),
        "irs_elements": irs_elements.tolist(),
        "obstacle": {
            "center": obstacle['center'].tolist(),
            "dimensions": obstacle['dimensions'].tolist()
        }
    }
    
    with open("data/environment.json", "w") as f:
        json.dump(env_data, f, indent=4)
    
    # Initialize lists to store all data
    all_data = []
    
    # Generate data for each episode
    for episode_id in tqdm(range(NUM_EPISODES), desc="Generating episodes"):
        # Generate random user trajectory for this episode
        user_positions, user_velocities = generate_user_trajectory(
            access_point, irs_elements, obstacle, episode_id)
        
        # Process each frame
        for frame_idx in range(FRAMES_PER_EPISODE):
            user_pos = user_positions[frame_idx]
            user_vel = user_velocities[frame_idx]
            
            # Check if direct path is blocked by obstacle
            direct_los_blocked = check_los(access_point, user_pos, obstacle)
            
            # Calculate direct channel
            h_direct = calculate_channel_coefficient(access_point, user_pos)
            if direct_los_blocked:
                # If blocked, severely attenuate the direct channel
                h_direct *= 0.1  # Significant attenuation
            
            # Calculate channels between Tx and each IRS element
            h_tx_irs = np.array([
                calculate_channel_coefficient(access_point, irs_elem, is_irs_path=True)
                for irs_elem in irs_elements
            ])
            
            # Calculate channels between each IRS element and Rx
            h_irs_rx = np.array([
                calculate_channel_coefficient(irs_elem, user_pos, is_irs_path=True)
                for irs_elem in irs_elements
            ])
            
            # Check if any IRS path is blocked
            irs_los_blocked = any(
                check_los(access_point, irs_elem, obstacle) or 
                check_los(irs_elem, user_pos, obstacle)
                for irs_elem in irs_elements
            )
            
            if irs_los_blocked:
                # If IRS path is blocked, attenuate the IRS channels
                h_tx_irs *= 0.1
                h_irs_rx *= 0.1
            
            # Calculate distances
            tx_rx_distance = calculate_distance(access_point, user_pos)
            tx_irs_distances = [calculate_distance(access_point, irs_elem) for irs_elem in irs_elements]
            irs_rx_distances = [calculate_distance(irs_elem, user_pos) for irs_elem in irs_elements]
            
            # Generate both random and optimal phase shifts
            random_phase_shifts = generate_random_phase_shifts()
            optimal_phase_shifts = generate_optimal_phase_shifts(h_tx_irs, h_irs_rx)
            
            # Calculate effective channels for both strategies
            effective_channel_random = calculate_effective_channel(
                h_direct, h_tx_irs, h_irs_rx, random_phase_shifts)
            
            effective_channel_optimal = calculate_effective_channel(
                h_direct, h_tx_irs, h_irs_rx, optimal_phase_shifts)
            
            # Calculate SNR for both strategies
            snr_db_random, received_power_dbm_random = calculate_snr(effective_channel_random)
            snr_db_optimal, received_power_dbm_optimal = calculate_snr(effective_channel_optimal)
            
            # Direct path SNR without IRS
            snr_db_direct, received_power_dbm_direct = calculate_snr(h_direct)
            
            # Determine best strategy based on SNR
            if snr_db_optimal > max(snr_db_random, snr_db_direct):
                best_strategy = "optimal"
                used_phase_shifts = optimal_phase_shifts
                effective_channel = effective_channel_optimal
                snr_db = snr_db_optimal
                received_power_dbm = received_power_dbm_optimal
            elif snr_db_random > snr_db_direct:
                best_strategy = "random"
                used_phase_shifts = random_phase_shifts
                effective_channel = effective_channel_random
                snr_db = snr_db_random
                received_power_dbm = received_power_dbm_random
            else:
                best_strategy = "direct"
                used_phase_shifts = np.zeros(NUM_IRS_ELEMENTS)  # No IRS used
                effective_channel = h_direct
                snr_db = snr_db_direct
                received_power_dbm = received_power_dbm_direct
            
            # Store data for this frame
            frame_data = {
                "realization_id": episode_id * FRAMES_PER_EPISODE + frame_idx,
                "episode_id": episode_id,
                "frame_id": frame_idx,
                "tx_rx_distance": tx_rx_distance,
                "tx_irs_distances": tx_irs_distances,
                "irs_rx_distances": irs_rx_distances,
                "user_position": user_pos.tolist(),
                "user_velocity": user_vel.tolist(),
                "direct_los_blocked": int(direct_los_blocked),
                "irs_los_blocked": int(irs_los_blocked),
                "H_direct_real": h_direct.real,
                "H_direct_imag": h_direct.imag,
                "H_tx_irs_real": [h.real for h in h_tx_irs],
                "H_tx_irs_imag": [h.imag for h in h_tx_irs],
                "H_irs_rx_real": [h.real for h in h_irs_rx],
                "H_irs_rx_imag": [h.imag for h in h_irs_rx],
                "irs_phase_shifts": used_phase_shifts.tolist(),
                "effective_channel_real": effective_channel.real,
                "effective_channel_imag": effective_channel.imag,
                "snr_db": snr_db,
                "received_power_dbm": received_power_dbm,
                "irs_strategy": best_strategy,
                "noise_power_dbm": NOISE_POWER_DBM,
                "tx_power_dbm": TX_POWER_DBM
            }
            
            all_data.append(frame_data)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(all_data)
    df.to_csv("data/irs_dataset.csv", index=False)
    print(f"Dataset generated: {len(all_data)} samples")
    
    # Save a sample for quick visualization
    sample_indices = np.random.choice(len(all_data), 5, replace=False)
    sample_data = [all_data[i] for i in sample_indices]
    with open("data/sample_data.json", "w") as f:
        json.dump(sample_data, f, indent=4)
    
    return df

if __name__ == "__main__":
    # Generate the dataset
    dataset = generate_dataset()
    
    # Print some statistics
    print("\nDataset Statistics:")
    print(f"Total samples: {len(dataset)}")
    print(f"Episodes: {NUM_EPISODES}")
    print(f"Frames per episode: {FRAMES_PER_EPISODE}")
    
    # Display distribution of strategies
    strategy_counts = dataset['irs_strategy'].value_counts()
    print("\nStrategy distribution:")
    for strategy, count in strategy_counts.items():
        print(f"  {strategy}: {count} ({count/len(dataset)*100:.2f}%)")
    
    # Display SNR statistics
    print(f"\nSNR statistics:")
    print(f"  Min: {dataset['snr_db'].min():.2f} dB")
    print(f"  Max: {dataset['snr_db'].max():.2f} dB")
    print(f"  Mean: {dataset['snr_db'].mean():.2f} dB")
    print(f"  Std: {dataset['snr_db'].std():.2f} dB")
    
    # Plot SNR distribution
    plt.figure(figsize=(10, 6))
    plt.hist(dataset['snr_db'], bins=50, alpha=0.7)
    plt.title("SNR Distribution in Dataset")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.savefig("data/snr_distribution.png")
    print("\nSNR distribution plot saved to data/snr_distribution.png")