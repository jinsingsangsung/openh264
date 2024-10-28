# read_pred.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import struct

MB_WIDTH = 16

class PredictionInputs:
    def __init__(self, data):
        # Read current block (16x16)
        self.current_block = np.frombuffer(data[0:256], dtype=np.uint8).reshape(MB_WIDTH, MB_WIDTH)
        
        # Read reference block (16x16)
        self.reference_block = np.frombuffer(data[256:512], dtype=np.uint8).reshape(MB_WIDTH, MB_WIDTH)
        
        # Read flags and positions
        self.has_left_ref = bool(data[512])
        self.mb_x = struct.unpack('i', data[513:517])[0]
        self.mb_y = struct.unpack('i', data[517:521])[0]
        self.is_iframe = bool(data[521])

def plot_blocks(inputs, block_num):
    """Plot current and reference blocks side by side"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot current block
    axes[0].imshow(inputs.current_block, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title(f'Current Block\nPosition: ({inputs.mb_x}, {inputs.mb_y})')
    axes[0].axis('off')
    
    # Plot reference block if it exists
    if inputs.has_left_ref:
        axes[1].imshow(inputs.reference_block, cmap='gray', vmin=0, vmax=255)
        axes[1].set_title('Reference Block (Left)')
    else:
        axes[1].set_title('No Reference Block')
    axes[1].axis('off')
    
    plt.suptitle(f'Macroblock #{block_num} - {"I" if inputs.is_iframe else "P"}-frame')
    plt.tight_layout()
    return fig

def read_bin_file(filename):
    """Read and display prediction inputs from binary file"""
    try:
        with open(filename, 'rb') as f:
            data = f.read()
            
        # Size of each prediction input struct
        struct_size = 522  # 256 + 256 + 1 + 4 + 4 + 1
        num_blocks = len(data) // struct_size
        
        print(f"Found {num_blocks} macroblocks in file")
        
        for i in range(num_blocks):
            start = i * struct_size
            end = start + struct_size
            block_data = data[start:end]
            
            inputs = PredictionInputs(block_data)
            
            # Create plot
            fig = plot_blocks(inputs, i)
            
            # Print information
            print(f"\n=== Macroblock #{i} ===")
            print(f"Position: ({inputs.mb_x}, {inputs.mb_y})")
            print(f"Frame type: {'I-frame' if inputs.is_iframe else 'P-frame'}")
            print(f"Has left reference: {'Yes' if inputs.has_left_ref else 'No'}")
            
            # Show plot
            plt.show()
            plt.close(fig)
            
            # Optional: ask to continue after each block
            if i < num_blocks - 1:
                input("Press Enter to see next macroblock...")
                
    except FileNotFoundError:
        print(f"Could not find file: {filename}")
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    # You can change this to match your .bin filename
    filenames = ["pred_inputs_p.bin", "pred_inputs_i4.bin", "pred_inputs_i16.bin"]
    
    for filename in filenames:
        if Path(filename).exists():
            print(f"\nReading {filename}:")
            read_bin_file(filename)