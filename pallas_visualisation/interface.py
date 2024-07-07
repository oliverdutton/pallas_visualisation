import gradio as gr
import tempfile
from .tooltip import create_tooltip
from .draw import draw_launch
import pandas as pd
import numpy as np

def launch(op_records, tensors, analysis_data=[], share=False, display_full_grid=False):
  grid_dims = np.array(list(op_records.keys())).max(0)+1

  cache = {}
  _, w, h = draw_launch(
    next(iter(op_records.values())), 
    tensors, 
    tempfile.NamedTemporaryFile(suffix=".svg").name
  )
  size = [w, h]
  height = 500 * size[1] / size[0]
  with gr.Blocks(
    css=".gradio-container button {overflow: auto} img.with-caption {height: %fpx !important; } .thumbnails { display: none; }  "
    % height
  ) as demo:
    with gr.Row():
      with gr.Column(scale=3, min_width=500):
        img = gr.Gallery(
          height=620,
          min_width=400,
          show_label=False,
          selected_index=0,
          preview=True,
          object_fit="scale-down",
        )
      with gr.Column(scale=1):
        program_id_sliders = [
          gr.Slider(0, dim-1, value=0, step=1, label=f"Grid dim {i}") for i, dim in enumerate(grid_dims)
        ]
        b1 = gr.Button("Precompute and show full grid")
        gr.Markdown("## Analysis")
        df = pd.DataFrame(analysis_data, columns=["Metric", "Value"])
        analysis_with_tooltip = create_tooltip(df)
        gr.HTML(analysis_with_tooltip)
        # if failures:
        #     gr.Markdown(
        #         show_label=False,
        #         value="## Invalid memory access in "
        #         + "\n * "
        #         + "\n* ".join(list(map(str, failures.keys()))),
        #     )

    def cache_block(idx):
      name = tempfile.NamedTemporaryFile(suffix=".svg")
      _, w, h = draw_launch(op_records[idx], tensors, name.name)
      size = [w, h]
      cache[idx] = (name, len(cache))

    def update(*inputs):
      idx = tuple(inputs)
      if idx not in cache:
        cache_block(idx)
        return gr.Gallery(
          value=[(cache[k][0].name, str(k)) for k in cache.keys()],
          selected_index=cache[idx][1],
          height=700,
        ), gr.Slider()
      return gr.Gallery(selected_index=cache[idx][1]), gr.Slider()

    def precompute_and_show(*inp):
      for idx in op_records.keys():
        if idx not in cache:
          cache_block(idx)
      return gr.Gallery(
        value=[(cache[k][0].name, str(k)) for k in cache.keys()],
        selected_index=None, #cache[tuple(inp)][1], 
      )
    for slider in program_id_sliders:
      slider.change(update, inputs=program_id_sliders, outputs=[img, b1], show_progress=False)
    b1.click(precompute_and_show, inputs=program_id_sliders, outputs=img, show_progress=True)
    demo.load(update, inputs=program_id_sliders, outputs=[img, b1]) 
    if (display_full_grid):
      demo.load(precompute_and_show, inputs=program_id_sliders, outputs=img)
  demo.launch(share=share, debug=False, height=800, quiet=True, show_api=False)  