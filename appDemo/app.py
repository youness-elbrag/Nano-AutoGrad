import gradio as gr
from demo import * 



article = """
<p style='text-align: center'>
    <a href='https://github.com/deep-matter/Nano-AutoGrad' target='_blank'>Github Repo Nano-AutoGrad</a>
</p>
"""

iface_webcam = gr.Interface(
    Optimization_training_progress_realtime,
    inputs=[
        gr.Radio(["Sparsity"], label="Task"),
        gr.Slider(minimum=1, maximum=100, label="Number of Epochs"),
        gr.Slider(minimum=0.01, maximum=1.0, label="Learning Rate"),
        gr.Number(label="Number of Layers"),
        gr.Number(label="Values for Weights")
        # gr.inputs.Slider(minimum=6, maximum=18, step=6, default=12),  # Leaving manual fps out for now
    ],
    outputs=[gr.Plot(),gr.Video(),gr.Video()],
    title="Optimization Training Progress",
    description="Real-time visualization of training progress",
    article=article,
    allow_flagging=False,
)

iface_file = gr.Interface(
    Optimization_training_progress_realtime,
    inputs=[
        gr.Radio(["Classification"], label="Task"),
        gr.Slider(minimum=1, maximum=100, label="Number of Epochs"),
        gr.Slider(minimum=0.01, maximum=1.0, label="Learning Rate"),
        gr.Number(label="Number of Layers"),
        gr.Number(label="Values for Weights")
    ],
    outputs=[gr.Plot(),gr.Video(),gr.Video()],
    title="Optimization Training Progress",
    description="Real-time visualization of training progress",
    article=article,
    allow_flagging=False,
)

if __name__ == '__main__':
    gr.TabbedInterface(
        interface_list=[iface_file, iface_webcam],
        tab_names=["Classification Task", "Sparsity Task"]
        
    ).launch(enable_queue=True)