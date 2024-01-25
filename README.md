# body2vec

idea: use your body to drive image generation

## components

### Pose Estimation

specs:
- at least 20 fps
- multiple people would be cool
- double check the output format needed for SDXL controlnet
  - https://huggingface.co/thibaud/controlnet-openpose-sdxl-1.0

inputs:
- cam stream

outputs:
- 224x244x3 image of poses, using the openpose format (need link for spec)
  - torch tensor on cuda device
  - fp32 or fp16
- vector of angles
  - fp32 or fp16
  - normalized to [0., 1.] range
  - torch tensor on cuda device
- list of names that map to the vector of angles
  - e.g. ["elbow_left", "elbow_right", ...]

### Tokenizer setup

- Select a prompt structure
  - e.g. "a blue horse" -> "a {color} {animal}"
- For each category
  - find other 1 token category members
  - get their embeddings
- Create a spherical interpolation setup

### SDXL-Turbo

- Attach controlnet
- Test compilation
- Write wrapping function

```python
class EmbeddingInterpolation


def run_step(
    pose_image: torch.Tensor,  # (1, 3, 224, 224)
    pose_angles: torch.Tensor,  # (1, 18)
)

```