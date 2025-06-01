import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import erfc
import argparse
import os

class IRSEnvironment:
    def __init__(self):
        """Initialize the IRS environment with default parameters."""
        # Default environment setup
        self.access_point = np.array([0.0, 0.0, 3.0])  # AP at origin, 3m height
        
        # Create IRS elements in a linear array
        self.num_elements = 8
        self.irs_spacing = 0.5  # 0.5m spacing between elements
        self.irs_elements = np.array([
            [5.0, i*self.irs_spacing - (self.num_elements-1)*self.irs_spacing/2, 2.0]
            for i in range(self.num_elements)
        ])
        
        # Add obstacle
        self.obstacle = {
            'center': np.array([2.5, 0.0, 1.5]),
            'dimensions': np.array([1.0, 2.0, 3.0])  # width, length, height
        }
        
        # Set carrier frequency and wavelength
        self.frequency = 2.4e9  # 2.4 GHz
        self.wavelength = 3e8 / self.frequency
    
    def calculate_optimal_phase_shifts(self, user_pos):
        """Calculate optimal phase shifts to maximize signal at user position."""
        irs_center = np.mean(self.irs_elements, axis=0)
        
        # Calculate angles for AP-IRS and IRS-User paths
        ap_to_irs = self.irs_elements - self.access_point
        irs_to_user = user_pos - self.irs_elements
        
        # Calculate total phase shift needed
        ap_phases = 2 * np.pi * np.linalg.norm(ap_to_irs, axis=1) / self.wavelength
        user_phases = 2 * np.pi * np.linalg.norm(irs_to_user, axis=1) / self.wavelength
        
        # Calculate optimal phase shifts to align phases
        total_phases = ap_phases + user_phases
        optimal_shifts = -np.mod(total_phases, 2*np.pi)
        
        return optimal_shifts

    def calculate_sinr(self, user_pos, phase_shifts, use_irs):
        """Calculate SINR for a given user position and phase shifts."""
        # Simplified SINR calculation
        distance = np.linalg.norm(user_pos - self.access_point)
        sinr_db = 30 - 20 * np.log10(distance)
        
        if use_irs:
            # IRS path
            irs_center = np.mean(self.irs_elements, axis=0)
            distance_irs = np.linalg.norm(user_pos - irs_center)
            sinr_db = 30 - 20 * np.log10(distance_irs)
        
        return sinr_db

class IRSVisualizer:
    def __init__(self, env):
        """Initialize the visualizer with environment data."""
        self.env = env
        
        # Create main figure for 3D environment and beam pattern
        plt.ion()
        self.main_fig = plt.figure(figsize=(14, 7))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 0.8])
        
        # 3D subplot for environment
        self.ax_3d = self.main_fig.add_subplot(gs[0], projection='3d')
        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')
        
        # Polar subplot for beam pattern
        self.ax_beam = self.main_fig.add_subplot(gs[1], projection='polar')
        self.ax_beam.set_theta_zero_location('E')
        self.ax_beam.set_theta_direction(-1)
        self.ax_beam.set_title('Beam Pattern')
        
        # Create separate figure for phase shifts
        self.phase_fig = plt.figure(figsize=(10, 6))
        self.ax_phase = self.phase_fig.add_subplot(111)
        self.ax_phase.set_xlabel('IRS Element')
        self.ax_phase.set_ylabel('Phase Shift (degrees)')
        self.ax_phase.set_title('IRS Phase Shifts')
        
        # Create separate figure for SINR vs BER
        self.sinr_ber_fig = plt.figure(figsize=(12, 8))
        self.ax_sinr_ber = self.sinr_ber_fig.add_subplot(111)
        self.ax_sinr_ber.set_xlabel('SINR (dB)', fontsize=14)
        self.ax_sinr_ber.set_ylabel('BER', fontsize=14)
        self.ax_sinr_ber.set_yscale('log')
        self.ax_sinr_ber.grid(True, which='both', linestyle='--', alpha=0.7)
        self.ax_sinr_ber.set_title('SINR vs BER for Different Modulation Schemes', fontsize=16)
        
        # Initialize SINR vs BER line
        self.sinr_ber_line, = self.ax_sinr_ber.plot([], [], 'bo-', linewidth=2)
        
        # Create separate figure for IRS elements vs SINR
        self.irs_sinr_fig = plt.figure(figsize=(10, 6))
        self.ax_elements = self.irs_sinr_fig.add_subplot(111)
        self.ax_elements.set_xlabel('Number of IRS Elements')
        self.ax_elements.set_ylabel('SINR (dB)')
        self.ax_elements.set_title('IRS Elements vs SINR')
        
        # Initialize IRS elements vs SINR line
        self.irs_sinr_line, = self.ax_elements.plot([], [], 'bo-', linewidth=2)
        
        # Calculate theoretical curves
        sinr_db = np.linspace(0, 30, 100)
        sinr_linear = 10**(sinr_db/10)
        
        # Theoretical curves
        ber_bpsk = 0.5 * erfc(np.sqrt(sinr_linear))
        ber_qpsk = 0.5 * erfc(np.sqrt(sinr_linear/2))
        ber_16qam = 3/4 * erfc(np.sqrt(3*sinr_linear/10))
        ber_64qam = 7/12 * erfc(np.sqrt(3*sinr_linear/26))
        
        # Plot the curves with improved styling
        self.ax_sinr_ber.plot(sinr_db, ber_bpsk, 'b-', linewidth=2, label='BPSK')
        self.ax_sinr_ber.plot(sinr_db, ber_qpsk, 'r--', linewidth=2, label='QPSK')
        self.ax_sinr_ber.plot(sinr_db, ber_16qam, 'g-.', linewidth=2, label='16-QAM')
        self.ax_sinr_ber.plot(sinr_db, ber_64qam, 'm:', linewidth=2, label='64-QAM')
        self.ax_sinr_ber.legend(fontsize=12, loc='upper right')
        self.ax_sinr_ber.set_ylim(1e-6, 1)
        
        # Initialize storage for history
        self.sinr_history = []
        self.ber_history = []
        
        # Initialize animation elements
        self.user_scatter = None
        self.beam_pattern = None
        self.user_direction = None
        self.signal_paths = {}
        
        # Initialize components
        self._init_components()
        
        # Adjust layout for all figures
        self.main_fig.tight_layout()
        self.phase_fig.tight_layout()
        self.sinr_ber_fig.tight_layout()
        
        # Position windows
        self.main_fig.canvas.manager.window.wm_geometry("+0+0")
        self.phase_fig.canvas.manager.window.wm_geometry("+0+600")
        self.sinr_ber_fig.canvas.manager.window.wm_geometry("+800+600")
        
    def _init_components(self):
        """Initialize static visualization components."""
        # Access Point
        self.ax_3d.scatter(*self.env.access_point, c='red', marker='^', s=200,
                          label='Access Point', edgecolor='black', linewidth=1)
        
        # IRS Panel
        irs_x = [element[0] for element in self.env.irs_elements]
        irs_y = [element[1] for element in self.env.irs_elements]
        irs_z = [element[2] for element in self.env.irs_elements]
        
        # Plot IRS elements
        self.ax_3d.scatter(irs_x, irs_y, irs_z, c='blue', marker='s', s=100,
                          label='IRS Elements', alpha=0.8)
        
        # Obstacle
        if hasattr(self.env, 'obstacle') and self.env.obstacle is not None:
            # Get obstacle parameters
            center = self.env.obstacle['center']
            dimensions = self.env.obstacle['dimensions']
            
            # Create obstacle box
            x_min = center[0] - dimensions[0]/2
            x_max = center[0] + dimensions[0]/2
            y_min = center[1] - dimensions[1]/2
            y_max = center[1] + dimensions[1]/2
            z_min = center[2] - dimensions[2]/2
            z_max = center[2] + dimensions[2]/2
            
            # Plot obstacle as a wireframe box
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            vertices = [
                [x_min, y_min, z_min], [x_max, y_min, z_min],
                [x_max, y_max, z_min], [x_min, y_max, z_min],
                [x_min, y_min, z_max], [x_max, y_min, z_max],
                [x_max, y_max, z_max], [x_min, y_max, z_max]
            ]
            faces = [
                [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
                [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
                [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
                [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
                [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
                [vertices[1], vertices[2], vertices[6], vertices[5]]   # right
            ]
            collection = Poly3DCollection(faces, alpha=0.25, linewidths=1, edgecolor='k')
            collection.set_facecolor('gray')
            self.ax_3d.add_collection3d(collection)
            self.ax_3d.text(x_min, y_min, z_max + 0.2, 'Obstacle', fontsize=10)
        
        # Initialize user position
        self.user_scatter = self.ax_3d.scatter([], [], [], c='green', s=100, 
                                             label='User', edgecolor='black', linewidth=1)
        
        # Initialize signal paths
        self.signal_paths = {
            'direct': self.ax_3d.plot([], [], [], 'g--', linewidth=2, alpha=0.7,
                                    label='Direct Path')[0],
            'irs': [self.ax_3d.plot([], [], [], 'b--', linewidth=1, alpha=0.5)[0]
                   for _ in range(len(self.env.irs_elements))]
        }
        
        # Add legend
        self.ax_3d.legend(loc='upper right')
        
        # Initialize beam pattern elements
        self.beam_pattern, = self.ax_beam.plot([], [], 'b-', linewidth=2, label='Beam Pattern')
        self.user_direction = self.ax_beam.scatter([], [], c='red', s=100, label='User Direction')
        
        # Add legend to beam pattern
        self.ax_beam.legend(loc='upper right')
        
        # Set beam pattern properties
        self.ax_beam.set_title('Beam Pattern')
        self.ax_beam.grid(True, alpha=0.3)
        
        # Initialize metrics text
        self.metrics_text = self.ax_3d.text2D(0.05, 0.95, '', transform=self.ax_3d.transAxes)
        
        # Initialize SINR vs BER plot
        self.sinr_history = []
        self.ber_history = []
        self.sinr_ber_line, = self.ax_sinr_ber.plot([], [], 'b-', linewidth=2)
        
        # Initialize phase shift plot
        self.phase_line, = self.ax_phase.plot([], [], 'bo-', linewidth=2)
        
        # Initialize IRS elements vs SINR plot
        self.irs_sinr_line, = self.ax_elements.plot([], [], 'bo-', linewidth=2)
        
    def animate_episode(self, episode_data):
        """Create animation for an episode."""
        num_frames = len(episode_data['user_positions'])
        
        def init():
            # Initialize user position
            self.user_scatter._offsets3d = ([0], [0], [0])
            
            # Initialize beam pattern
            self.beam_pattern.set_data([], [])
            
            # Initialize user direction marker
            self.user_direction.set_offsets(np.zeros((1, 2)))
            
            # Initialize direct path
            self.signal_paths['direct'].set_data_3d([], [], [])
            
            # Initialize IRS paths
            for path in self.signal_paths['irs']:
                path.set_data_3d([], [], [])
            
            # Initialize metrics text
            self.metrics_text.set_text('')
            
            # Initialize SINR vs BER plot
            self.sinr_ber_line.set_data([], [])
            
            # Initialize phase shift plot
            self.phase_line.set_data([], [])
            
            # Initialize IRS elements vs SINR plot
            self.irs_sinr_line.set_data([], [])
            
            return (self.user_scatter, self.beam_pattern, self.user_direction,
                   self.signal_paths['direct'], *self.signal_paths['irs'],
                   self.metrics_text, self.sinr_ber_line, self.phase_line,
                   self.irs_sinr_line)
        
        def update(frame):
            # Update user position
            user_pos = np.array(episode_data['user_positions'][frame])
            self.user_scatter._offsets3d = ([user_pos[0]], [user_pos[1]], [user_pos[2]])
            
            # Update signal paths
            # Check if direct path is blocked by obstacle
            is_blocked = False
            if hasattr(self.env, '_is_blocked') and hasattr(self.env, 'obstacle') and self.env.obstacle is not None:
                is_blocked = self.env._is_blocked(self.env.access_point, user_pos)
            
            # Update direct path
            self.signal_paths['direct'].set_data_3d(
                [self.env.access_point[0], user_pos[0]],
                [self.env.access_point[1], user_pos[1]],
                [self.env.access_point[2], user_pos[2]]
            )
            
            # Set direct path style based on blockage
            if is_blocked:
                self.signal_paths['direct'].set_linestyle(':')
                self.signal_paths['direct'].set_alpha(0.3)
            else:
                self.signal_paths['direct'].set_linestyle('--')
                self.signal_paths['direct'].set_alpha(0.7)
            
            # Update IRS paths
            phase_shifts = episode_data['phase_shifts'][frame]
            for i, path in enumerate(self.signal_paths['irs']):
                # Calculate path from AP to IRS element
                irs_pos = self.env.irs_elements[i]
                path.set_data_3d(
                    [self.env.access_point[0], irs_pos[0], user_pos[0]],
                    [self.env.access_point[1], irs_pos[1], user_pos[1]],
                    [self.env.access_point[2], irs_pos[2], user_pos[2]]
                )
                # Set color based on phase shift
                phase_contribution = (1 + np.cos(phase_shifts[i])) / 2
                path.set_alpha(0.3 + 0.7 * phase_contribution)
            
            # Update beam pattern
            angles = np.linspace(0, 2*np.pi, 360)
            pattern = self._calculate_radiation_pattern(phase_shifts, angles)
            self.beam_pattern.set_data(angles, pattern)
            
            # Update user direction on beam pattern
            irs_center = np.mean(self.env.irs_elements, axis=0)
            user_direction = user_pos - irs_center
            user_angle = np.arctan2(user_direction[1], user_direction[0])
            if user_angle < 0:
                user_angle += 2*np.pi
            self.user_direction.set_offsets([[user_angle, 1.0]])
            
            # Update metrics text
            sinr_db = episode_data['sinr_db'][frame] if 'sinr_db' in episode_data else 0
            self.metrics_text.set_text(f'SINR: {sinr_db:.1f} dB')
            
            # Update SINR vs BER plot
            self.ax_sinr_ber.plot(sinr_db, episode_data['ber'][frame], 'bo')
            self.sinr_history.append(sinr_db)
            self.ber_history.append(episode_data['ber'][frame])
            self.sinr_ber_line.set_data(self.sinr_history, self.ber_history)
            
            # Update phase shift plot
            self.phase_line.set_data(range(self.env.num_elements), phase_shifts)
            
            return (self.user_scatter, self.beam_pattern, self.user_direction,
                   self.signal_paths['direct'], *self.signal_paths['irs'],
                   self.metrics_text, self.sinr_ber_line, self.phase_line,
                   self.irs_sinr_line)
        
        # Create animation
        self.anim = FuncAnimation(
            self.main_fig, update, init_func=init,
            frames=num_frames, interval=50,
            blit=True, repeat=True
        )
        
        # Show all plots and block until windows are closed
        plt.ioff()
        plt.show()
        
    def _calculate_radiation_pattern(self, phase_shifts, angles):
        """Calculate radiation pattern based on phase shifts."""
        wavelength = 3e8 / 2.4e9  # 2.4 GHz carrier frequency
        k = 2 * np.pi / wavelength
        d = self.env.irs_spacing  # Element spacing
        
        # Calculate array factor
        array_factor = np.zeros_like(angles, dtype=complex)
        for i, angle in enumerate(angles):
            # Phase difference between elements
            phase_diff = k * d * np.sin(angle)
            # Sum contributions from all elements
            element_contributions = np.exp(1j * (np.arange(self.env.num_elements) * phase_diff + phase_shifts))
            array_factor[i] = np.sum(element_contributions)
        
        # Convert to power pattern and normalize
        pattern = np.abs(array_factor)
        return pattern / np.max(pattern)

def main():
    """Main function to run the visualization."""
    print("Initializing environment...")
    env = IRSEnvironment()
    
    print("Creating visualizer...")
    visualizer = IRSVisualizer(env)
    
    print("Generating sample episode data...")
    # Generate sample episode data
    num_steps = 100
    episode_data = {
        'user_positions': [],
        'phase_shifts': [],
        'sinr_db': [],
        'ber': []
    }
    
    # Generate circular trajectory
    center = np.array([5.0, 0.0, 1.5])
    radius = 3.0
    angles = np.linspace(0, 2*np.pi, num_steps)
    
    for angle in angles:
        # User position
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        z = center[2]
        user_pos = np.array([x, y, z])
        
        # Calculate optimal phase shifts
        phase_shifts = env.calculate_optimal_phase_shifts(user_pos)
        
        # Calculate SINR (simplified)
        distance = np.linalg.norm(user_pos - env.access_point)
        sinr_db = 30 - 20 * np.log10(distance)
        
        # Calculate BER (simplified)
        ber = 0.5 * erfc(np.sqrt(10**(sinr_db/10)))
        
        # Store data
        episode_data['user_positions'].append(user_pos)
        episode_data['phase_shifts'].append(phase_shifts)
        episode_data['sinr_db'].append(sinr_db)
        episode_data['ber'].append(ber)
    
    print("Starting animation...")
    visualizer.animate_episode(episode_data)
    
    print("Animation complete. Close the plot window to exit.")

if __name__ == "__main__":
    print("Starting IRS Beamforming Visualization...")
    main()
