import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from IPython.display import clear_output
from core.engine import Value
from sklearn.datasets import make_moons, make_blobs
import plotly.graph_objects as go
import plotly.io as pio
import imageio
from IPython.display import Image, display

clear_output()

# Iterate through the first 6 images

def extract_path_df(path_dir, index_show):
    path_file = []

    for filesname in os.listdir(path_dir):
        path_file.append(os.path.join(path_dir,filesname))

    for data_df in range(0,len(path_file)):
        data_frame = pd.read_csv(path_file[data_df])
        show_df = data_frame.head(index_show)
    return path_file , f"dataframe: {show_df}"

def loading_df_to_numpy(path_file):
    data_df = pd.read_csv(path_file)
    data = np.array(data_df) 
    m, n = data.shape
    data_train = data[1000:m].T
    Y_train = data_train[0][40900:]
    X_train = data_train[1:n][:, 40900:]


    # Manually split the data into training and testing sets
    split_index = int(X_train.shape[0] * 0.8)  # 80% for training, 20% for testing
    X_train_split = X_train[:, :10]
    Y_train_split = Y_train[:10]
    X_test_split = X_train[:,:5]
    Y_test_split = Y_train[:5]

    return X_train_split, X_test_split, Y_train_split, Y_test_split

def initialize_data(n_samples: int, noise: float):
    input_data, Target = make_moons(n_samples=n_samples, noise=noise)

    Target = Target * 2 - 1  # make y be -1 or 14
    # fig.close()
    return input_data, Target


def plot_sample(DATA_TRAIN,DATA_LABEL):
    num_images = min(6, DATA_TRAIN.shape[1])
    fig, axs = plt.subplots(2, 3, figsize=(10, 7))

    for i in range(num_images):
        label = DATA_LABEL
        image = DATA_TRAIN[:, i]
        current_image = image.reshape(28, 28) * 255

        # Determine the subplot coordinates
        row = i // 3
        col = i % 3

    # Plot the image in the corresponding subplot
        axs[row, col].imshow(current_image, cmap='gray')
        axs[row, col].set_title("Label: {}".format(label))
        axs[row, col].axis('off')

    plt.tight_layout()
    plt.savefig("sample.png")

def copy(model):
  model = SparseMLP(nin=2, nouts=[16, 16, 1], sparsities=[0.,0.9,0.8])
  model.parameters()
  return model

def Zvals(model , X_train):
  global X
  h = 0.25
  x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
  y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                      np.arange(y_min, y_max, h))
  Xmesh = np.c_[xx.ravel(), yy.ravel()]
  inputs = [list(map(Value, xrow)) for xrow in Xmesh]
  scores = list(map(model, inputs))
  Z = np.array([s.data > 0 for s in scores])
  Z = Z.reshape(xx.shape)
  return Z


def dboundary(model):
  global X
  global Y
  h = 0.25
  x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                      np.arange(y_min, y_max, h))
  Xmesh = np.c_[xx.ravel(), yy.ravel()]
  fig, ax = plt.subplots(figsize=(8,8))
  Z = Zvals(model)
  ln = ax.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
  ax.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap=plt.cm.Spectral)
  ax.set_xlim(xx.min(), xx.max())
  ax.set_ylim(yy.min(), yy.max())
  return  fig,ax,ln


def graph_trace(Path, nframes, interval):
    fig = go.Figure()

    def add_frame(index):
        frame_path = f"assets/{Path}_{index}.png"
        fig.add_layout_image(
            source=frame_path,
            xref="x",
            yref="y",
            x=0,
            y=1,
            sizex=1,
            sizey=1,
            sizing="contain",
            opacity=1,
            layer="below"
        )

    for i in range(nframes):
        add_frame(i)

    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        width=800,
        height=600,
        autosize=False,
        hovermode=False,
        updatemenus=[dict(type="buttons", showactive=False)],
    )

    animation_path = "out/Graph.mp4"
    with imageio.get_writer(animation_path, mode="I") as writer:
        for i in range(nframes):
            frame_path = f"assets/{Path}_{i}.png"
            image = imageio.imread(frame_path)
            writer.append_data(image)

    # Display the animation
    image_bytes = pio.to_image(fig, format="png")
    display(Image(image_bytes))
