import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')

LABEL_NAMES  = ['AAC', 'Body', 'Face', 'Gestures', 'Look', 'Vocalization']

def load_data(base_path):
    """Load all the .npy files from a specific directory"""
    print(f"Loading data files from: {base_path}")
    data = {}
    
    # Load gradient files
    for i in range(6):
        try:
            file_path = os.path.join(base_path, f'label{i}_grads.npy')
            data[f'label{i}_grads'] = np.load(file_path)
            print(f"  Loaded label{i}_grads.npy - shape: {data[f'label{i}_grads'].shape}")
        except FileNotFoundError:
            print(f"  Warning: label{i}_grads.npy not found")
            data[f'label{i}_grads'] = None
    
    return data

def normalize_min_max(arr):
    """Min-max normalization"""
    if arr.size == 0 or np.isnan(arr).all():
        return np.nan
    arr_min = np.nanmin(arr)
    arr_max = np.nanmax(arr)
    if arr_max == arr_min:
        return np.zeros_like(arr)
    return (arr - arr_min) / (arr_max - arr_min)

def process_gradients(data):
    """Process and normalize gradient data"""
    print("  Processing gradients...")
    normalized_data = {}
    
    for i in range(6):
        key = f'label{i}_grads'
        if data[key] is not None:
            mean_grads = data[key].mean(0)
            normalized = normalize_min_max(mean_grads)
            normalized_data[f'label{i}_norm'] = normalized
            
            if isinstance(normalized, np.ndarray) and normalized.size > 0:
                print(f"  Normalized {LABEL_NAMES[i]} gradients - shape: {normalized.shape}")
            else:
                print(f"  Warning: {LABEL_NAMES[i]} gradients resulted in invalid normalization")
                normalized_data[f'label{i}_norm'] = None
        else:
            normalized_data[f'label{i}_norm'] = None
    
    return normalized_data

def create_body_heatmap(values, label_name, ax=None):
    """Create body pose heatmap visualization for upper body keypoints only"""
    if values is None or (isinstance(values, float) and np.isnan(values)):
        print(f"Cannot create heatmap for {label_name}: invalid values")
        return None
    
    if not isinstance(values, np.ndarray) or values.size < 25:
        print(f"Cannot create heatmap for {label_name}: insufficient data points (need 25, got {values.size if isinstance(values, np.ndarray) else 0})")
        return None
    
    all_positions = {
        0: (0.0, -0.6), 1: (0.0, -0.2), 2: (0.0, 0.5), 3: (0.0, 0.7),
        4: (-0.2, 0.5), 5: (-0.5, 0.3), 6: (-0.7, 0.0), 7: (-0.8, -0.1),
        8: (0.2, 0.5), 9: (0.5, 0.3), 10: (0.7, 0.0), 11: (0.8, -0.1),
        12: (-0.15, -0.6), 13: (-0.2, -1.0), 14: (-0.25, -1.4), 15: (-0.3, -1.5),
        16: (0.15, -0.6), 17: (0.2, -1.0), 18: (0.25, -1.4), 19: (0.3, -1.5),
        20: (0.0, 0.15), 21: (-0.9, -0.2), 22: (-0.7, 0.1), 23: (0.9, -0.2), 24: (0.7, 0.1)
    }
    
    # Upper body keypoints only (exclude 0, 12-19 which are hip and legs)
    upper_body_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 20, 21, 22, 23, 24]
    
    positions = {idx: all_positions[idx] for idx in upper_body_indices}
    
    # Upper body joint pairs only
    joint_pairs = [
        [1, 20], [20, 2], [2, 3],  # Central spine/neck
        [2, 4], [4, 5], [5, 6], [6, 7],  # Left arm
        [6, 22], [7, 21],  # Left hand connections
        [2, 8], [8, 9], [9, 10], [10, 11],  # Right arm
        [10, 24], [11, 23]  # Right hand connections
    ]
    
    # Get coordinates and values for upper body keypoints
    x_coords = [positions[i][0] for i in upper_body_indices]
    y_coords = [positions[i][1] for i in upper_body_indices]
    upper_body_values = [values[i] for i in upper_body_indices]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = None
    
    scatter_plot = ax.scatter(
        x_coords, y_coords, 
        c=upper_body_values, cmap='coolwarm', s=200, vmin=0, vmax=1, alpha=1.0
    )
    
    for (start, end) in joint_pairs:
        if start in positions and end in positions:
            x1, y1 = positions[start]
            x2, y2 = positions[end]
            ax.plot([x1, x2], [y1, y2], linewidth=2, color='black', zorder=-1)
    
    if fig is not None:
        cbar = plt.colorbar(scatter_plot, label='Node Strength', ax=ax)
        cbar.set_label('Node Strength', fontsize=20)
        cbar.ax.tick_params(labelsize=16)
        # Set colorbar ticks to only show 0, 0.5, and 1
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(['0', '0.5', '1'])

    ax.set_title(f"{label_name}", fontsize=24, pad=20)
    ax.axis('equal')
    ax.axis('off')
    # Set axis limits to focus on upper body only
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.25, 0.75)
    
    print(f"Created {label_name} heatmap")
    return fig

def create_consolidated_visualization(data, normalized_data, output_path):
    """Create a single figure with all heatmap visualizations"""
    print("  Creating consolidated visualization...")
    
    valid_heatmaps = []
    
    for i in range(6):
        key = f'label{i}_norm'
        if key in normalized_data and normalized_data[key] is not None:
            if isinstance(normalized_data[key], np.ndarray) and normalized_data[key].size >= 25:
                valid_heatmaps.append((i, LABEL_NAMES[i]))
    
    n_heatmaps = len(valid_heatmaps)
    
    if n_heatmaps == 0:
        print("  No valid visualizations to create")
        return False
    
    fig = plt.figure(figsize=(6 * n_heatmaps, 4))
    scatter_plot = None # To store the mappable object for the colorbar
    
    for i, (idx, label_name) in enumerate(valid_heatmaps):
        ax = plt.subplot(1, n_heatmaps, i + 1)
        values = normalized_data[f'label{idx}_norm']
        
        # Original positions for all 25 keypoints
        all_positions = {
            0: (0.0, -0.6), 1: (0.0, -0.2), 2: (0.0, 0.5), 3: (0.0, 0.7),
            4: (-0.2, 0.5), 5: (-0.5, 0.3), 6: (-0.7, 0.0), 7: (-0.8, -0.1),
            8: (0.2, 0.5), 9: (0.5, 0.3), 10: (0.7, 0.0), 11: (0.8, -0.1),
            12: (-0.15, -0.6), 13: (-0.2, -1.0), 14: (-0.25, -1.4), 15: (-0.3, -1.5),
            16: (0.15, -0.6), 17: (0.2, -1.0), 18: (0.25, -1.4), 19: (0.3, -1.5),
            20: (0.0, 0.15), 21: (-0.9, -0.2), 22: (-0.7, 0.1), 23: (0.9, -0.2), 24: (0.7, 0.1)
        }
        
        # Upper body keypoints only (exclude 0, 12-19 which are hip and legs)
        upper_body_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 20, 21, 22, 23, 24]
        
        # Filter positions for upper body only
        positions = {idx: all_positions[idx] for idx in upper_body_indices}
        
        # Upper body joint pairs only
        joint_pairs = [
            [1, 20], [20, 2], [2, 3],  # Central spine/neck
            [2, 4], [4, 5], [5, 6], [6, 7],  # Left arm
            [6, 22], [7, 21],  # Left hand connections
            [2, 8], [8, 9], [9, 10], [10, 11],  # Right arm
            [10, 24], [11, 23]  # Right hand connections
        ]
        
        # Get coordinates and values for upper body keypoints
        x_coords = [positions[j][0] for j in upper_body_indices]
        y_coords = [positions[j][1] for j in upper_body_indices]
        upper_body_values = [values[j] for j in upper_body_indices]
        
        scatter_plot = ax.scatter(x_coords, y_coords, c=upper_body_values, cmap='coolwarm', s=200, vmin=0, vmax=1)
        
        for start, end in joint_pairs:
            if start in positions and end in positions:
                x1, y1 = positions[start]
                x2, y2 = positions[end]
                ax.plot([x1, x2], [y1, y2], linewidth=2, color='black', zorder=-1)
        
        ax.set_title(f"{label_name}", fontsize=30)
        ax.axis('equal')
        ax.axis('off')
        # Set axis limits to focus on upper body only
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-0.25, 0.75)
    
    if scatter_plot and n_heatmaps > 0:
        # Adjust subplots to make room for the colorbar
        fig.subplots_adjust(right=0.92)
        # Define position of the colorbar axis: [left, bottom, width, height]
        # The 'width' (3rd value) is reduced to make the colorbar thinner.
        cbar_ax = fig.add_axes([0.93, 0.15, 0.008, 0.7])
        cbar = fig.colorbar(scatter_plot, cax=cbar_ax)
        cbar.set_label('Node Strength', fontsize=20)
        cbar.ax.tick_params(labelsize=16)
        # Set colorbar ticks to only show 0, 0.5, and 1
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(['0', '0.5', '1'])

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved consolidated visualization with {n_heatmaps} plots to {output_path}")
    return True

def process_subdirectory(subdir_path, output_dir):
    """Process a single subdirectory and create visualization"""
    subdir_name = os.path.basename(subdir_path)
    gradients_path = os.path.join(subdir_path, 'gradients')
    
    if not os.path.exists(gradients_path):
        print(f"\nSkipping {subdir_name}: no gradients folder found")
        return False
    
    print(f"\nProcessing subdirectory: {subdir_name}")
    print("=" * 40)
    
    data = load_data(gradients_path)
    normalized_data = process_gradients(data)
    
    output_path = os.path.join(output_dir, f"{subdir_name}.pdf")
    success = create_consolidated_visualization(data, normalized_data, output_path)
    
    return success

def main():
    """Main execution function"""
    print("Pose Visualization Script - Batch Processing")
    print("=" * 60)
    
    # Base paths
    base_dir = 'gradients'
    output_dir = 'images'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Get all subdirectories in individual_results
    try:
        subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        subdirs.sort()  # Sort for consistent processing order
        print(f"\nFound {len(subdirs)} subdirectories to process")
    except Exception as e:
        print(f"Error accessing directory: {e}")
        return
    
    # Process each subdirectory
    successful = 0
    failed = 0
    
    for subdir in subdirs:
        subdir_path = os.path.join(base_dir, subdir)
        success = process_subdirectory(subdir_path, output_dir)
        
        if success:
            successful += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print(f"Successfully processed: {successful}")
    print(f"Failed/Skipped: {failed}")
    print(f"Total: {len(subdirs)}")
    print(f"\nVisualizations saved to: {output_dir}")

if __name__ == "__main__":
    main()