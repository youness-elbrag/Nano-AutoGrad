import random
import numpy as np
from sklearn.datasets import make_moons, make_blobs
from core.engine import Value
from core.nn import Neuron, Layer, MLP
from core.SparsityNN import SparseMLP
import time 
from functools import partial
import matplotlib.animation as animation
import shutil
from IPython.display import HTML

import pandas as pd
import plotly.subplots as sp
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import os
from uilit import *

# make up a dataset
def initialize_data(n_samples: int, noise: float):
    input_data, Target = make_moons(n_samples=n_samples, noise=noise)
    print(input_data[0][0])
    print(Target[0])

    Target = Target * 2 - 1  # make y be -1 or 14
    fig = go.Figure(data=go.Scatter(x=input_data[:, 0], y=input_data[:, 1], mode='markers', marker=dict(color=Target, size=10, colorscale='jet')))
    fig.write_image('plot.png')
    # fig.close()
    return input_data, Target

# loss function
def loss(model,X_train , y_train , batch_size=None):
    
    # inline DataLoader :)
    if batch_size is None:
        Xb, yb = X_train, y_train
    else:
        ri = np.random.permutation(X_train.shape[0])[:batch_size]
        Xb, yb = X_train[ri], y_train[ri]

    inputs = [list(map(Value, xrow)) for xrow in Xb]
    
    # forward the model to get scores
    scores = list(map(model, inputs))
    
    # svm "max-margin" loss
    losses = [(1 + -yi*scorei).relu() for yi, scorei in zip(yb, scores)]
    data_loss = sum(losses) * (1.0 / len(losses))
    # L2 regularization
    alpha = 0.05 
    reg_loss = alpha * sum((p*p for p in model.parameters()))
    total_loss = data_loss + reg_loss
    
    # also get accuracy
    accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(yb, scores)]
    return total_loss, sum(accuracy) / len(accuracy)

def Optimization_training_progress_realtime(num_epoch, model, x_train, y_train):
    # Create empty lists for loss and accuracy
    loss_data = []
    accuracy_data = []
    Wieghts_parameters = []


    # Create subplots with shared x-axis
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Loss", "Accuracy"))

    # Initialize empty lists for loss and accuracy
    loss_data = []
    accuracy_data = []

    # Create initial empty traces
    loss_trace = go.Scatter(x=[], y=[], mode="lines", name="Loss")
    accuracy_trace = go.Scatter(x=[], y=[], mode="lines", name="Accuracy")

    # Add initial traces to the subplots
    fig.add_trace(loss_trace, row=1, col=1)
    fig.add_trace(accuracy_trace, row=2, col=1)
     # Update layout
    fig.update_layout(
        title="Training Progress",
        xaxis_title="Epoch",
        yaxis=dict(title="Loss"),
        yaxis2=dict(title="Accuracy"),
        showlegend=False,
        hovermode='x',
        height=800,
        width=800,
        template='plotly_white'
    )
    # Define animation frames
    frames = []

    for k in range(num_epoch):
        # Forward pass
        total_loss, acc = loss(model, x_train, y_train, batch_size=None)

        # Backward pass
        model.zero_grad()
        total_loss.backward()

        # Update (SGD)
        learning_rate = 1.0 - 0.9 * k / 100
        for p in model.parameters():
            p.data -= learning_rate * p.grad    

        Wieghts_parameters.append(model)

        # Append data to lists
        loss_data.append(total_loss.data)
        accuracy_data.append(acc)

         # Update traces
        with fig.batch_update():
            fig.data[0].x = list(range(k+1))
            fig.data[0].y = loss_data
            fig.data[1].x = list(range(k+1))
            fig.data[1].y = accuracy_data

        # Append current frame to frames list
        frames.append(go.Frame(data=[fig.data[0], fig.data[1]]))

    # Add frames to animation
    fig.frames = frames

    # Create animation
    animation = go.Figure(fig)

    # Set animation settings
    animation.update_layout(
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 500, "redraw": True},
                                        "fromcurrent": True, "transition": {"duration": 0}}],
                        "label": "Play",
                        "method": "animate"
                    }
                ],
                "showactive": False,
                "type": "buttons"
            }
        ]
    )

    # Display animation
    # Save animation as GIF
    animation.show()
    animation.write_image("training_animation.svg", engine="kaleido")
    return Wieghts_parameters

def create_animation(models, X_train, y_train):
    fig = go.Figure()

    # Generate contour data for each model
    h = 0.25
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Xmesh = np.c_[xx.ravel(), yy.ravel()]

    Z_values = []
    for model in models:
        inputs = [list(map(Value, xrow)) for xrow in Xmesh]
        scores = list(map(model, inputs))
        Z = np.array([s.data > 0 for s in scores])
        Z = Z.reshape(xx.shape)
        print(Z.shape)
        Z_values.append(Z)


    # Create initial contour trace
    contour_trace = go.Contour(
        x=xx.ravel(),
        y=[].reshape(xx.shape),
        colorscale='Jet',
        showscale=False,
        opacity=0.8
    )
    fig.add_trace(contour_trace)

    # Scatter plot
    scatter_trace = go.Scatter(
        x=X_train[:, 0],
        y=X_train[:, 1],
        mode='markers',
        marker=dict(
            color=y_train,
            colorscale='Jet',
            size=6,
            showscale=False
        )
    )
    fig.add_trace(scatter_trace)

    # Layout settings
    fig.update_layout(
        title="Model Visualization",
        xaxis=dict(range=[x_min, x_max]),
        yaxis=dict(range=[y_min, y_max]),
        showlegend=False,
        width=800,
        height=600,
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white')
    )

    # Animation frames
    frames = []
    for Z in Z_values:
        contour_trace = go.Contour(
            x=xx.ravel(),
            y=yy.ravel(),
            z=Z,
            colorscale='Jet',
            showscale=False,
            opacity=0.8
        )
        frame = go.Frame(data=[contour_trace])
        frames.append(frame)

    fig.frames = frames

    # Animation layout settings
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    )
                ],
                showactive=False,
                direction="left",
                x=0.1,
                y=0,
                xanchor="right",
                yanchor="bottom"
            ),
        ],
    )

    fig.show()


        

if __name__ == "__main__":

    np.random.seed(1337)
    random.seed(1337)
    X_train , y_train = initialize_data(n_samples=100 ,noise=0.1)
    model = MLP(2, [16,16,1]) # 2-layer neural network
    # model = SparseMLP(nin=2, nouts=[16, 16, 1], sparsities=[0.,0.9,0.8]) # 2-layer neural network
    print("number of parameters", len(model.parameters()))
    models = Optimization_training_progress_realtime(num_epoch=30,model=model , x_train=X_train ,y_train=y_train)
    create_animation(models, X_train, y_train)

    

    

