import lunar_tools as lt
import numpy as np
from pythonosc.dispatcher import Dispatcher
from pythonosc import osc_server
import threading
import math
from lunar_tools.control_input import MidiInput
from copy import deepcopy

# Create server and client
IP = "10.20.16.145"
OSC_IP = "10.20.17.122"
ZMQP = 5559
OSC_PORT = 5557

client = lt.ZMQPairEndpoint(is_server=False, ip=IP, port=ZMQP)

# midi controller
akai_lpd8 = MidiInput(device_name="akai_midimix")

#%%
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, StableDiffusionXLControlNetPipeline
import torch

from diffusers import AutoencoderTiny
from sfast.compilers.stable_diffusion_pipeline_compiler import (compile, CompilationConfig)
from diffusers import ControlNetModel
from diffusers.utils import load_image
import lunar_tools as lt
import numpy as np
from PIL import Image
from PIL import Image
import requests
import numpy as np
import cv2

from src.modulation import Modulator

device = torch.device("cuda:0") 

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

use_maxperf = False 

print('Using pure text')
pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")

pipe.to(device)
pipe.vae = AutoencoderTiny.from_pretrained('madebyollin/taesdxl', torch_device=device, torch_dtype=torch.float16)
pipe.vae = pipe.vae.to(device)
pipe.set_progress_bar_config(disable=True)

text_encoder_1_copy = deepcopy(pipe.text_encoder)
text_encoder_2_copy = deepcopy(pipe.text_encoder)

if use_maxperf:
    config = CompilationConfig.Default()
    config.enable_xformers = True
    config.enable_triton = True
    config.enable_cuda_graph = True
    config.enable_jit = True
    config.enable_jit_freeze = True
    config.trace_scheduler = True
    config.enable_cnn_optimization = True
    config.preserve_parameters = False
    config.prefer_lowp_gemm = True
    
    pipe = compile(pipe, config)

print("starting warmup")
pipe(
    prompt="a profession in a bg background",
    image=Image.new(mode="RGB", size=(200, 200)),
    negative_prompt="nsfw",
)
print("warmup done")

# pipe.text_encoder = text_encoder_1_copy
# pipe.text_encoder_2 = text_encoder_2_copy

torch.manual_seed(1)
noise_level = 0

categories = {
    "color": ["red", "green", "blue"],
    "profession": ["diver", "farmer", "king"],
    "bg": ["jungle", "snow", "city"],
    "animal": ["mouse", "dolphin", "lion"],
    "age": ["adult", "old"],
    "gender": ["male", "female"],
}
modulator = Modulator(
    categories,
    pipe.tokenizer,
    pipe.text_encoder,
    pipe.text_encoder_2,
)
cntr = 0

right_elbow = 0

class VarHolder():
    def __init__(self):
        self.osc_data = [0.0] * 5
        self.mins = [0.0] * 5
        self.maxs = [0.0] * 5
    

var_holder = VarHolder()

def launch_server():
    print("test")
    def write_to_global_buffer(*args):
        try:
            print(args)
            for i in range(5):
                maybe = float(args[i+1])
                if math.isnan(maybe):
                    assert False
                if i == 4: # gesture
                    var_holder.osc_data[4] = maybe
                    continue
                if maybe < var_holder.mins[i]:
                    var_holder.mins[i] = maybe
                if maybe > var_holder.maxs[i]:
                    var_holder.maxs[i] = maybe

                delta = max(var_holder.maxs[i]-var_holder.mins[i], 0.001)
                normalized = (maybe-var_holder.mins[i])/delta
                
                clamped = max(min(normalized, 1.0), 0.0)
                var_holder.osc_data[i] = clamped * 0.2 + var_holder.osc_data[i] * 0.8
        except Exception as e:
            print(e)

    dispatcher = Dispatcher()
    dispatcher.map("/tracker", write_to_global_buffer)
    comms_server = osc_server.ThreadingOSCUDPServer(
        (OSC_IP, OSC_PORT), dispatcher)
    print("serving forever")
    comms_server.serve_forever()
    print("sent")

server_thread = threading.Thread(target=launch_server)
server_thread.start()

generator = torch.Generator(device=device)

cap = cv2.VideoCapture(0)  # Initialize the webcam capture
ret, frame = cap.read()
print(f"frame_shape: {frame.shape}")
height, width = frame.shape[:2]
renderer = lt.Renderer(width=width*2, height=height*2)

gesture_target = "profession"
gesture_cooldown = 20
prompt = "a profession in a bg background"

try:
    cntr = 0
    frame_buffer = None
    while True:
        decay = akai_lpd8.get("A0", val_min=0, val_max=1, val_default=.1)
        blend = akai_lpd8.get("A1", val_min=0, val_max=1, val_default=.1)
        seed = akai_lpd8.get("A2", val_min=0, val_max=20, val_default=16)

        generator.manual_seed(int(seed))

        modulator.set_idx_embeddings(
            "bg",
            var_holder.osc_data[0],
            decay,
        )

        modulator.set_idx_embeddings(
            gesture_target,
            var_holder.osc_data[1],
            decay,
        )

        modulator.set_idx_embeddings(
            "age",
            var_holder.osc_data[2],
            decay,
        )

        modulator.set_idx_embeddings(
            "gender",
            var_holder.osc_data[3],
            decay,
        )

        if var_holder.osc_data[4] != 0 and gesture_cooldown < 0:
            if gesture_target == "profession":
                gesture_target = "animal"
                prompt="an anthropomorphic age gender animal in a bg background"
            else:
                gesture_target = "profession"
                prompt = "a age gender profession in a bg background"
            gesture_cooldown = 50

        gesture_cooldown -= 1

        # modulator.set_idx_embeddings(
        #     "bg",
        #     gamma,
        # )

        ret, frame = cap.read()
        # Convert the BGR image to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if frame_buffer is not None:
            frame = cv2.addWeighted(frame, 1-blend, frame_buffer, blend, 0)

        image = pipe(
            prompt=prompt,
            negative_prompt="nsfw, naked, blurry",
            image=frame/255,
            guidance_scale=0.0, 
            num_inference_steps=2,
            strength=.5,
            output_type="np",
            generator=generator,
        ).images[0]

        image = np.uint8(image*255)

        frame_buffer = image

        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.flip(image, 1)
        cv2.putText(image, f"{var_holder.osc_data[0]:.2f}", (10,30), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 1)
        cv2.putText(image, f"{var_holder.osc_data[1]:.2f}", (10,60), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 1)
        cv2.putText(image, f"{var_holder.osc_data[2]:.2f}", (10,90), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 1)
        cv2.putText(image, f"{var_holder.osc_data[3]:.2f}", (10,120), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 1)
        cv2.putText(image, f"{var_holder.osc_data[4]:.2f}", (10,150), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 1)


        cv2.putText(image, f"decay: {decay:.2f}", (10,180), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 1)
        cv2.putText(image, f"blend: {blend:.2f}", (10,210), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 1)
        cv2.putText(image, f"seed: {seed:.2f}", (10,240), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 1)

        renderer.render(image)

        # cv2.imshow('Image', image)

        # Break the loop with 'q'
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        # client.send_img(image)  
finally:
    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()