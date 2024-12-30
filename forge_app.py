import gradio as gr
import numpy as np
import random

import spaces
from models import SwittiPipeline
import torch
import gc


pipe = SwittiPipeline.from_pretrained()
spaces.automatically_move_pipeline_components(pipe)

MAX_SEED = np.iinfo(np.int32).max


@spaces.GPU(duration=65)
def infer(
    prompt,
    negative_prompt="",
    seed=42,
    randomize_seed=False,
    guidance_scale=4.0,
    top_k=400,
    top_p=0.95,
    more_smooth=True,
    smooth_start_si=2,
    turn_off_cfg_start_si=10,
    more_diverse=True,
    last_scale_temp=1,
    progress=gr.Progress(track_tqdm=True),
):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    
    turn_on_cfg_start_si = 2 if more_diverse else 0

    image = pipe(
        prompt=prompt,
        null_prompt=negative_prompt,
        cfg=guidance_scale,
        top_p=top_p,
        top_k=top_k,
        more_smooth=more_smooth,
        smooth_start_si=smooth_start_si,
        turn_off_cfg_start_si=turn_off_cfg_start_si,
        turn_on_cfg_start_si=turn_on_cfg_start_si,
        seed=seed,
        last_scale_temp=last_scale_temp,
    )[0]

    return image, seed

def unload():
    global pipe
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


examples = [
    "Cute winter dragon baby, kawaii, Pixar, ultra detailed, glacial background, extremely realistic.", 
    "A cosmonaut under the starry sky in a purple radiation zone against the background of huge Amanita mushrooms in the style of dark botanical",
    "A small house on a mountain top",
    "A lighthouse in a giant wave, origami style.",
    "The Mandalorian by masamune shirow, fighting stance, in the snow, cinematic lighting, intricate detail, character design",
    "Sci-fi cosmic diarama of a quasar and jellyfish in a resin cube, volumetric lighting, high resolution, hdr, sharpen, Photorealism",   
]

css = """
#col-container {
    margin: 0 auto;
    max-width: 640px;
}
"""

with gr.Blocks(css=css) as demo:
    with gr.Row():
        with gr.Column(elem_id="col-container"):
            with gr.Row():
                gr.Markdown(" # [Switti](https://yandex-research.github.io/switti)")

            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                lines=3,
                placeholder="Enter your prompt",
                container=False,
            )

            seed = gr.Number(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                value=0,
            )

            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

            guidance_scale = gr.Slider(
                label="Guidance scale",
                minimum=1.0,
                maximum=10.,
                step=0.1,
                value=4.5,
            )

            with gr.Accordion("Advanced Settings", open=False):
                negative_prompt = gr.Text(
                    label="Negative prompt",
                    max_lines=1,
                    placeholder="Enter a negative prompt",
                    visible=True,
                )

                with gr.Row():
                    top_k = gr.Slider(
                        label="Sampling top k",
                        minimum=10,
                        maximum=1000,
                        step=10,
                        value=400,
                    )
                    top_p = gr.Slider(
                        label="Sampling top p",
                        minimum=0.0,
                        maximum=1.,
                        step=0.01,
                        value=0.95,
                    )
                    
                with gr.Row():
                    more_smooth = gr.Checkbox(label="Smoothing with Gumbel softmax sampling", value=True)
                    smooth_start_si = gr.Slider(
                        label="Smoothing starting scale",
                        minimum=0,
                        maximum=10,
                        step=1,
                        value=2,
                    )
                    turn_off_cfg_start_si = gr.Slider(
                        label="Disable CFG starting scale",
                        minimum=0,
                        maximum=10,
                        step=1,
                        value=8,
                    )
                with gr.Row():
                    more_diverse = gr.Checkbox(label="More diverse", value=False)
                    last_scale_temp = gr.Slider(
                        label="Temperature after disabling CFG",
                        minimum=0.1,
                        maximum=10,
                        step=0.1,
                        value=0.1,
                    )

        with gr.Column(elem_id="col-container"):
            run_button = gr.Button("Run", variant="primary")
            result = gr.Image(label="Result", show_label=False)
            gr.Examples(examples=examples, inputs=[prompt], outputs=[result, seed], fn=infer, cache_examples=False)# cache_mode="lazy")


    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn=infer,
        inputs=[
            prompt,
            negative_prompt,
            seed,
            randomize_seed,
            guidance_scale,
            top_k,
            top_p,
            more_smooth,
            smooth_start_si,
            turn_off_cfg_start_si,
            more_diverse,
            last_scale_temp,
        ],
        outputs=[result, seed],
    )

    demo.unload(fn=unload)

if __name__ == "__main__":
    demo.launch()
    