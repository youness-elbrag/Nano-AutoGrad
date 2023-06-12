import random
import numpy as np
from sklearn.datasets import make_moons, make_blobs
from core.engine import Value
from core.nn import Neuron, Layer, MLP
import time 
import shutil
import pandas as pd
import plotly.subplots as sp
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import os


# make up a dataset
def initialize_data(n_samples: int, noise: float):
    input_data, Target = make_moons(n_samples=n_samples, noise=noise)
    Target = Target * 2 - 1  # make y be -1 or 1
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
    alpha = 1e-4
    reg_loss = alpha * sum((p*p for p in model.parameters()))
    total_loss = data_loss + reg_loss
    
    # also get accuracy
    accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(yb, scores)]
    return total_loss, sum(accuracy) / len(accuracy)

def Optimazition_training_progress_realtime(num_epoch, model, x_train, y_train):
    filename = f"assets/real_training_{num_epoch-1}.png"
    if os.path.exists(filename):
        shutil.rmtree('assets/')
        os.makedirs('assets/')
    
    # Create subplots with shared x-axis
    fig = go.FigureWidget(make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Loss", "Accuracy")))

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
    nframes = num_epoch
    interval = num_epoch * 2

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

        # Update traces
        with fig.batch_update():
            fig.data[0].x = list(range(k+1))
            fig.data[0].y = loss_data
            fig.data[1].x = list(range(k+1))
            fig.data[1].y = accuracy_data

            # Save plot as image
            fig.write_image(f"assets/real_training_{k}.png")

        # Append loss and accuracy to the lists
        loss_data.append(total_loss.data)
        accuracy_data.append(acc)
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
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=40, cmap=plt.cm.Spectral)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.savefig(f'assets/plot_res_{k}.png')

    fig_1 = plt.figure()
    def animate_predicion(i):
        im1 = plt.imread(f"assets/plot_res_{i}.png")
        plt.imshow(im1)
        plt.title(f"Epoch: {i+1}\nLoss: {loss_data[i]:.4f} - Accuracy: {accuracy_data[i]:.4f}")
        plt.xlabel("prediction")
        fig_1.tight_layout()

    anim_= FuncAnimation(fig_1, animate_predicion, frames=nframes, interval=interval)
    anim_.save("out/training.gif", writer="imagemagick")

    fig_2 = plt.figure()
    def animate_training(i):
        im2 = plt.imread(f"assets/real_training_{i}.png")
        plt.imshow(im2)
        plt.title(f"Epoch: {i+1}\nLoss: {loss_data[i]:.4f} - Accuracy: {accuracy_data[i]:.4f}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        fig_2.tight_layout()

    anim = FuncAnimation(fig_2, animate_training, frames=nframes, interval=interval)
    anim.save("out/Prediction.gif", writer="imagemagick")
    return model

        

if __name__ == "__main__":

    np.random.seed(1337)
    random.seed(1337)
    ## initiale the data 
    X_train , y_train = initialize_data(n_samples=100 ,noise=0.1)

    # initialize a model 
    model = MLP(2, [16, 16, 1]) # 2-layer neural network
    print(model)
    print("number of parameters", len(model.parameters()))
    
    model = Optimazition_training_progress_realtime(num_epoch=50,model=model , x_train=X_train ,y_train=y_train)
    time.sleep(0.5)

    # Visualize decision boundary

    time.sleep(0.5)
    

