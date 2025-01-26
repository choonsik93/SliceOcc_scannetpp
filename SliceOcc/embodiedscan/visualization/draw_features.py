import numpy as np
import matplotlib.pyplot as plt
import torch

def visualize_feature_map(feature_map, mode='sum', channel_indices=None, output_dir=None):
    if mode == 'sum':
        summed_feature_map = np.sum(feature_map, axis=1)[0]
        plt.imshow(summed_feature_map, cmap='viridis')
        plt.title('Summed Feature Map')
        plt.colorbar()
        plt.savefig(f'{output_dir}/summed_feature_map_9.png')
        
    elif mode == 'sample':
        if channel_indices is None:
            raise ValueError("Please provide a list of channel indices for 'sample' mode")
        for channel_index in channel_indices:
            sampled_feature_map = feature_map[0, channel_index]
            plt.imshow(sampled_feature_map, cmap='viridis')
            plt.title(f'Sampled Feature Map (Channel {channel_index})')
            plt.colorbar()
            plt.savefig(f'{output_dir}/sampled_feature_map_channel_{channel_index}.png')
            
    else:
        raise ValueError("Mode should be either 'sum' or 'sample'")


if __name__ == "__main__":
    feat = torch.load("/mnt/data/ljn/code/EmbodiedScan/vis/vis_feat/scannet_scene0269_01/9.pth")
    feature_map = feat.numpy()
    visualize_feature_map(feature_map, mode='sum', output_dir="/mnt/data/ljn/code/EmbodiedScan/embodiedscan/visualization/features/")
    #visualize_feature_map(feature_map, mode='sample', channel_indices=[0, 32, 64, 128, 256])