import imageio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import numpy as np

def tensors_to_gif(tensor_list, gif_path, total_duration, T, grid_size=16, nr_channels=1):
    images = []

    for tensor in tensor_list:
        # Ensure tensor is detached and converted to CPU
        tensor = tensor.view(grid_size, grid_size, nr_channels)
        tensor = tensor.detach().cpu()

        # Convert to numpy array
        image_array = tensor.numpy()

        # # Scale the image array to [0, 255] if needed
        # if image_array.min() < 0 or image_array.max() > 1:
        #     image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())

        # Convert to 8-bit grayscale
        image_array = (image_array * 255).astype(np.uint8)

        # If the image is single-channel, remove the last dimension
        if nr_channels == 1:
            image_array = image_array.squeeze(-1)

        # Add to list of images
        images.append(image_array)
        
        # plt.imshow(image_array, interpolation='none')
        # plt.show()

    nr_frames = len(tensor_list)
    print(nr_frames, T)
    print('nr_frames=', nr_frames, 'saving to: ', gif_path)
    imageio.mimsave(gif_path, images, duration=total_duration/nr_frames)

# example usage
# tensors_to_gif(intermediates_arr[1], 'output.gif', total_duration=1, grid_size=16, nr_channels=1)

# gif_path = 'output.gif'
# gif = imageio.mimread(gif_path)

# # Create a figure
# fig = plt.figure()

# # Function to update the frame
# def update_frame(frame):
#     plt.imshow(frame, interpolation='none', cmap='gray')
#     plt.axis('off')

# # Create an animation
# ani = animation.FuncAnimation(fig, update_frame, frames=gif, repeat=True)

# # Display the animation
# plt.show()