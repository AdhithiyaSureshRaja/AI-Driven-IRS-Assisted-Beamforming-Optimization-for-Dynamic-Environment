"""
Dataset Visualizer for IRS Environment

This script visualizes the generated IRS dataset with:
- 3D visualization of user movement, IRS panel, access point, and obstacle
- Dotted lines showing signal path (either direct or IRS-reflected)
- Ability to select and visualize any episode from the dataset

References:
- Visualization approaches based on standard practices in IRS literature (Wu et al., 2021)
- Signal paths and channel models from Wu & Zhang, 2019
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import json
import argparse

def load_environment():
    """Load the environment setup from the saved file"""
    with open("data/environment.json", "r") as f:
        env_data = json.load(f)
    
    access_point = np.array(env_data["access_point"])
    irs_elements = np.array(env_data["irs_elements"])
    obstacle = {
        "center": np.array(env_data["obstacle"]["center"]),
        "dimensions": np.array(env_data["obstacle"]["dimensions"])
    }
    
    return access_point, irs_elements, obstacle

def load_episode_data(episode_id):
    """Load data for a specific episode from the dataset"""
    # Load the full dataset
    df = pd.read_csv("data/irs_dataset.csv")
    
    # Filter data for the requested episode
    episode_data = df[df["episode_id"] == episode_id].copy()
    
    if len(episode_data) == 0:
        raise ValueError(f"Episode {episode_id} not found in dataset")
    
    # Sort by frame_id to ensure correct order
    episode_data = episode_data.sort_values("frame_id")
    
    return episode_data

def create_obstacle_mesh(obstacle):
    """Create mesh data for visualizing the obstacle as a cuboid"""
    center = obstacle["center"]
    dimensions = obstacle["dimensions"]
    
    # Calculate the 8 vertices of the cuboid
    half_dim = dimensions / 2
    vertices = np.array([
        center + np.array([half_dim[0], half_dim[1], half_dim[2]]),
        center + np.array([half_dim[0], half_dim[1], -half_dim[2]]),
        center + np.array([half_dim[0], -half_dim[1], half_dim[2]]),
        center + np.array([half_dim[0], -half_dim[1], -half_dim[2]]),
        center + np.array([-half_dim[0], half_dim[1], half_dim[2]]),
        center + np.array([-half_dim[0], half_dim[1], -half_dim[2]]),
        center + np.array([-half_dim[0], -half_dim[1], half_dim[2]]),
        center + np.array([-half_dim[0], -half_dim[1], -half_dim[2]])
    ])
    
    # Define the 6 faces using indices of vertices
    faces = [
        [vertices[0], vertices[1], vertices[3], vertices[2]],
        [vertices[4], vertices[5], vertices[7], vertices[6]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[0], vertices[2], vertices[6], vertices[4]],
        [vertices[1], vertices[3], vertices[7], vertices[5]]
    ]
    
    return faces

def setup_3d_plot():
    """Set up the 3D plotting environment"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title('IRS Communication Environment Visualization')
    
    # Set reasonable view limits based on environment scale
    ax.set_xlim(-10, 60)
    ax.set_ylim(-30, 30)
    ax.set_zlim(0, 20)
    
    # Set a reasonable viewing angle
    ax.view_init(elev=20, azim=-35)
    
    return fig, ax

def animate_episode(episode_id=0):
    """Create an animation of the specified episode"""
    # Load environment and episode data
    access_point, irs_elements, obstacle = load_environment()
    episode_data = load_episode_data(episode_id)
    
    # Setup 3D plot
    fig, ax = setup_3d_plot()
    
    # Elements to be updated in each frame
    user_trail, = ax.plot([], [], [], 'r-', alpha=0.5, linewidth=1)
    user_point, = ax.plot([], [], [], 'ro', markersize=8)
    signal_path_direct, = ax.plot([], [], [], 'g--', linewidth=1, alpha=0.7)
    signal_path_irs = []
    for _ in range(len(irs_elements)):
        line, = ax.plot([], [], [], 'b--', linewidth=1, alpha=0.5)
        signal_path_irs.append(line)
    
    # Plot static elements
    # Access point (base station)
    ax.plot([access_point[0]], [access_point[1]], [access_point[2]], 'bs', markersize=10, label='Access Point')
    
    # IRS elements
    irs_x = [element[0] for element in irs_elements]
    irs_y = [element[1] for element in irs_elements]
    irs_z = [element[2] for element in irs_elements]
    ax.scatter(irs_x, irs_y, irs_z, c='cyan', marker='^', s=100, label='IRS Elements')
    
    # IRS panel outline (connect the elements)
    min_z = min(irs_z)
    max_z = max(irs_z)
    panel_corners = [
        [irs_x[0]-0.5, irs_y[0]-0.5, min_z-0.5],
        [irs_x[0]-0.5, irs_y[0]-0.5, max_z+0.5],
        [irs_x[0]-0.5, irs_y[0]+0.5, max_z+0.5],
        [irs_x[0]-0.5, irs_y[0]+0.5, min_z-0.5]
    ]
    panel = Poly3DCollection([panel_corners], alpha=0.2, color='cyan')
    ax.add_collection3d(panel)
    
    # Obstacle
    faces = create_obstacle_mesh(obstacle)
    obstacle_mesh = Poly3DCollection(faces, alpha=0.25, color='gray', linewidths=1, edgecolors='k')
    ax.add_collection3d(obstacle_mesh)
    
    # Add legend
    ax.legend(loc='upper right')
    
    def init():
        user_trail.set_data([], [])
        user_trail.set_3d_properties([])
        user_point.set_data([], [])
        user_point.set_3d_properties([])
        signal_path_direct.set_data([], [])
        signal_path_direct.set_3d_properties([])
        for line in signal_path_irs:
            line.set_data([], [])
            line.set_3d_properties([])
        return (user_trail, user_point, signal_path_direct, *signal_path_irs)
    
    def update(frame):
        # Get data for this frame
        frame_data = episode_data.iloc[frame]
        user_pos = np.array(eval(frame_data['user_position']))
        
        # Update user trail (show recent positions)
        start_frame = max(0, frame - 10)  # Show last 10 positions
        trail_data = episode_data.iloc[start_frame:frame+1]
        trail_x = [eval(pos)[0] for pos in trail_data['user_position']]
        trail_y = [eval(pos)[1] for pos in trail_data['user_position']]
        trail_z = [eval(pos)[2] for pos in trail_data['user_position']]
        user_trail.set_data(trail_x, trail_y)
        user_trail.set_3d_properties(trail_z)
        
        # Update current user position
        user_point.set_data([user_pos[0]], [user_pos[1]])
        user_point.set_3d_properties([user_pos[2]])
        
        # Determine signal path based on strategy
        strategy = frame_data['irs_strategy']
        
        # Direct path (always show but with different style based on usage)
        if strategy == 'direct':
            signal_path_direct.set_data([access_point[0], user_pos[0]], [access_point[1], user_pos[1]])
            signal_path_direct.set_3d_properties([access_point[2], user_pos[2]])
            signal_path_direct.set_color('g')
            signal_path_direct.set_linestyle('-')
            signal_path_direct.set_alpha(1.0)
            signal_path_direct.set_linewidth(2.0)
        else:
            signal_path_direct.set_data([access_point[0], user_pos[0]], [access_point[1], user_pos[1]])
            signal_path_direct.set_3d_properties([access_point[2], user_pos[2]])
            signal_path_direct.set_color('gray')
            signal_path_direct.set_linestyle('--')
            signal_path_direct.set_alpha(0.3)
            signal_path_direct.set_linewidth(1.0)
        
        # IRS paths
        if strategy in ['random', 'optimal']:
            # Get phase shifts
            phase_shifts = eval(frame_data['irs_phase_shifts'])
            
            # Draw lines from AP to each IRS element and from IRS elements to user
            for i, (irs_element, line) in enumerate(zip(irs_elements, signal_path_irs)):
                # Color intensity based on phase shift contribution
                phase_contribution = (1 + np.cos(phase_shifts[i])) / 2  # Normalize to 0-1
                
                # Draw line from AP to IRS element
                line.set_data([access_point[0], irs_element[0], user_pos[0]], 
                              [access_point[1], irs_element[1], user_pos[1]])
                line.set_3d_properties([access_point[2], irs_element[2], user_pos[2]])
                line.set_color('blue')
                line.set_alpha(0.7 * phase_contribution + 0.3)  # Alpha based on contribution
                line.set_linewidth(1.5)
        else:
            # Clear IRS paths if not used
            for line in signal_path_irs:
                line.set_data([], [])
                line.set_3d_properties([])
        
        # Add strategy and SNR text
        ax.set_title(f'Episode {episode_id}, Frame {frame}, Strategy: {strategy.upper()}, SNR: {frame_data["snr_db"]:.2f} dB')
        
        return (user_trail, user_point, signal_path_direct, *signal_path_irs)
    
    # Create animation
    frames = len(episode_data)
    interval = 200  # ms between frames
    
    ani = FuncAnimation(fig, update, frames=frames, init_func=init, 
                        interval=interval, blit=False)
    
    plt.tight_layout()
    return fig, ani

def visualize_episode(episode_id=0, save_animation=False):
    """Visualize a specific episode with optional animation saving"""
    fig, ani = animate_episode(episode_id)
    
    if save_animation:
        # Save as MP4
        animation_path = f"visualizations/episode_{episode_id}_animation.mp4"
        ani.save(animation_path, writer='ffmpeg', fps=10, dpi=200)
        print(f"Animation saved to {animation_path}")
    
    plt.show()

def visualize_sinr_comparison(episode_id=0):
    """
    Visualize SINR comparison between direct and IRS paths for a given episode
    Based on SNR analysis in Wu & Zhang, 2019 and Lin et al., 2020
    """
    # Load episode data
    episode_data = load_episode_data(episode_id)
    
    # Extract frame indices and SNR values
    frames = episode_data['frame_id'].values
    snr_values = episode_data['snr_db'].values
    strategies = episode_data['irs_strategy'].values
    
    # Separate SNR by strategy
    snr_direct = np.where(strategies == 'direct', snr_values, np.nan)
    snr_random = np.where(strategies == 'random', snr_values, np.nan)
    snr_optimal = np.where(strategies == 'optimal', snr_values, np.nan)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(frames, snr_values, 'k-', alpha=0.5, label='Actual SNR')
    plt.plot(frames, snr_direct, 'go', alpha=0.7, label='Direct Path')
    plt.plot(frames, snr_random, 'bv', alpha=0.7, label='Random IRS')
    plt.plot(frames, snr_optimal, 'r^', alpha=0.7, label='Optimal IRS')
    
    plt.xlabel('Frame')
    plt.ylabel('SNR (dB)')
    plt.title(f'SNR Comparison for Episode {episode_id}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"visualizations/episode_{episode_id}_snr_comparison.png")
    plt.show()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Visualize IRS environment dataset")
    parser.add_argument("--episode", type=int, default=0, help="Episode ID to visualize")
    parser.add_argument("--save", action="store_true", help="Save animation to file")
    parser.add_argument("--snr", action="store_true", help="Show SNR comparison plot")
    args = parser.parse_args()
    
    # Create visualizations directory if it doesn't exist
    import os
    os.makedirs("visualizations", exist_ok=True)
    
    # Visualize the requested episode
    visualize_episode(args.episode, args.save)
    
    # Show SNR comparison if requested
    if args.snr:
        visualize_sinr_comparison(args.episode)

if __name__ == "__main__":
    main()