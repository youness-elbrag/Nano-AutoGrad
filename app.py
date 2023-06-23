import gradio as gr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from demo import * 



# Define the input and output types for Gradio interface
task_input = gr.inputs.Radio(["Sparsity", "Classification"], label="Task")
num_epoch_input = gr.inputs.Slider(minimum=1, maximum=100, default=50, label="Number of Epochs")
learning_rate_input = gr.inputs.Slider(minimum=0.01, maximum=1.0, default=0.1, label="Learning Rate")
num_layer_input = gr.inputs.Number(label="Number of Layers")
values_weights_input = gr.inputs.Number(label="Values for Weights")


outputs = [gr.Plot(),gr.Plot()]
# Create the Gradio interface
iface = gr.Interface(
    fn=Optimization_training_progress_realtime,
    inputs=[task_input, num_epoch_input, learning_rate_input, num_layer_input, values_weights_input],
    outputs=outputs,
    title="Optimization Training Progress",
    description="Real-time visualization of training progress",
)

iface.launch(share=True)
