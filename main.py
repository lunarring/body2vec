import lunar_tools as lt
import numpy as np
from pythonosc.dispatcher import Dispatcher
from pythonosc import osc_server
import threading
import math

# Create server and client
IP = "10.20.16.145"
OSC_IP = "10.20.17.122"
ZMQP = 5559
OSC_PORT = 5557

client = lt.ZMQPairEndpoint(is_server=False, ip=IP, port=ZMQP)

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

device = torch.device("cuda:1") 

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

torch.manual_seed(1)
noise_level = 0

categories = {
    "color": ["red", "green", "blue"],
    "animal": ["cat", "dog", "mouse", "eagle", "shark", "worm", "cow", "dinosaur"],
    "bg": ["jungle", "snow", "city"]
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
        self.osc_data = 0.

var_holder = VarHolder()

def launch_server():
    print("test")
    def write_to_global_buffer(*args):
        try:
            maybe = float(args[-1])
            if math.isnan(maybe):
                assert False
            print(maybe)
            right_elbow = maybe/180.
            right_elbow = max(min(right_elbow, 1.0), 0.0)
            var_holder.osc_data = right_elbow * 0.2 + var_holder.osc_data * 0.8
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

try:
    cntr = 0
    cap = cv2.VideoCapture(0)  # Initialize the webcam capture
    while True:
        alpha = var_holder.osc_data
        # beta = (cntr % 35) / 35
        # gamma = (cntr % 55) / 55
        cntr += 0.1

        modulator.set_idx_embeddings(
            "color",
            alpha,
        )

        # modulator.set_idx_embeddings(
        #     "animal",
        #     beta,
        # )

        # modulator.set_idx_embeddings(
        #     "bg",
        #     gamma,
        # )

        ret, frame = cap.read()
        # Convert the BGR image to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame / 255.0

        image = pipe(
            prompt="a color astronaut",
            image=frame,
            guidance_scale=0.0, 
            num_inference_steps=2,
            strength=.5,
            output_type="np",
        ).images[0]
            
        # Render the image

        # add angle to the image with cv2

        image = np.asanyarray(image*255)
        image = np.uint8(image)
        client.send_img(image)  
finally:
    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()