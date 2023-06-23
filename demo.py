import random
import numpy as np
from core.engine import Value
from core.nn import Neuron, Layer, MLP
from core.Graph import draw_dot

import time 
from graphviz import Digraph
import imageio
from functools import partial
import matplotlib.animation as animation
import shutil
from IPython.display import HTML
import os
import pandas as pd
import plotly.subplots as sp
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation ,FFMpegWriter
import matplotlib.pyplot as plt
from uilit import *
os.environ["PATH"] += os.pathsep + 'app/lib/Python 3.11/site-packages/graphviz'

path_data = 'digit-recognizer/data/'



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
    return total_loss, sum(accuracy) / len(accuracy) ,scores

def Optimization_training_progress_realtime(Task,num_epoch, learning_rate ,num_layer,values_wieghts):
    filename = f"assets/plot_res_{num_epoch-1}.png"
    if os.path.exists(filename):
        shutil.rmtree('assets/')
        os.makedirs('assets/')
    # Create empty lists for loss and accuracy
    loss_data = []
    accuracy_data = []
    model = MLP(int(num_layer), [int(values_wieghts),int(values_wieghts),1]) # 2-layer neural network
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
        height=500,
        width=500,
        template='plotly_white'
    )
    # Define animation frames
    frames = []
    if Task in "Sparsity":
        X_train , Y_train = initialize_data(n_samples=100 ,noise=0.1)
        
    elif Task in "Classification":
        FILES , _  = extract_path_df(path_data,10)
        X_train, X_test, Y_train, Y_test = loading_df_to_numpy(FILES[0])
    for k in range(int(num_epoch)):
        # Forward pass
        total_loss, acc,scores = loss(model, X_train, Y_train, batch_size=None)

        # Backward pass
        model.zero_grad()
        total_loss.backward()
        draw_dot(model(scores),f'graph_wights_update_{k}')

        # Update (SGD)

        learning_rate = 1.0 - 0.9 * k / 100

        for p in model.parameters():
            p.data -= learning_rate * p.grad 

        if k % 2 == 0:
            print(f"step {k} loss {total_loss.data}, accuracy {acc*100}%")
   
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

        if Task in "Sparsity":
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

            plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
            plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, s=40, cmap=plt.cm.Spectral)
            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())
            plt.savefig(f'assets/plot_res_{k}.png')
     # Add frames to animation
    fig.frames = frames
    nframes = int(num_epoch)
    interval = int(num_epoch) * 2

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
        
    if Task in "Sparsity":
        graph_trace("graph_wights_update", nframes,interval)
        fig_2 = plt.figure()
        def animate_predicion(i):
            im1 = plt.imread(f"assets/plot_res_{i}.png")
            plt.imshow(im1)
            plt.title(f"Epoch: {i+1}\nLoss: {loss_data[i]:.4f} - Accuracy: {accuracy_data[i]:.4f}")
            plt.xlabel("prediction")
            fig_2.tight_layout()

        anim_= FuncAnimation(fig_2, animate_predicion, frames=nframes, interval=interval)
        anim_.save("out/training.gif", writer="imagemagick")
        # Read the animated GIF
       # Create the Plotly figure
        img = "out/training.gif"
        # Display the animation
     # Show the figure
        return animation , fig_2

    if Task in "Classification":
        # graph_trace("graph_wights_update", nframes,interval)
        inputs_test = [list(map(Value, xrow)) for xrow in X_test]
        predictions = [scorei.data.argmax() for scorei in list(map(model, inputs_test))]

        # Plot a few examples
        num_examples = 8
        fig_1, axes = plt.subplots(4, 4, figsize=(10, 10))
        fig_1.subplots_adjust(hspace=0.4, wspace=0.4)
        def animate(i):
            for i, ax in enumerate(axes.flatten()):
                ax.imshow(X_test[:, i,None ], cmap="gray")
                ax.set_title(f"Predicted: {Y_test[i]}")
                ax.axis('off')
        fig_1.tight_layout()
        lin_ani = FuncAnimation(fig_1, animate, frames=len(X_test), interval=200)
        FFwriter = FFMpegWriter(fps=10)

        lin_ani.save('out/Predicted.mp4', writer=FFwriter)
        # fig_1.savefig("reulst.png")
        return animation ,fig_1 



   

if __name__ == "__main__":

    np.random.seed(1337)
    random.seed(1337)
    models = Optimization_training_progress_realtime(
        Task="Classification",num_epoch=4, learning_rate=0.002 ,
        num_layer=2,values_wieghts=4)

    

    

