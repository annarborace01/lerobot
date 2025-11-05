## Torch Installation Notes:
- For RTX6000 blackwell, it requires torch at least 2.71 + cu128
```
pip install torch==2.7.1 torchvision==0.22.1 \
-extra-index-url https://download.pytorch.org/whl/cu128 
```

- torchcodex seems not working for ffmpeg 6 somehow
```
conda install ffmpeg=7.1.1 -c conda-forge
```

## Flow matching

Basically, we want to find a match from gaussian noise to the targeted distribution so that 
we can sample from the gaussian noise to get the targeted sample.

At training stage, we train a model for predicting the flow velocity: 
```
# ===========================
# Goal: Learn a velocity field v_θ(x_t, t) that maps noisy samples back to data
# 
# Notation:
#   x₀ ~ p_data     : clean data sample from the dataset
#   x₁ ~ p_prior    : noise sample (standard Gaussian)
#   t ∈ [0, 1]      : interpolation time (0 = data, 1 = noise)
#   x_t = (1-t)x₀ + tx₁ : linear interpolation (flow path)
#   v_t = x₁ - x₀   : target velocity field (direction from data to noise)
#   v_θ(x_t, t)     : learned velocity field (neural network)

for x in dataset:
    z = torch.randn_like(x)        # x₁ ~ N(0, I): sample noise from prior
    t = random.uniform(0, 1)       # t ~ U(0, 1): sample random time
    xt = (1-t) * x + t * z         # x_t = (1-t)x₀ + tx₁: flow interpolation
    v = z - x                      # v_t = x₁ - x₀: target velocity (constant for linear flow)
    vt = model(xt, t, conditioned_inputs...)  # v_θ(x_t, t, c): predict velocity
    loss = F.mse(v - vt)           # L = ||v_t - v_θ(x_t, t)||²: regression loss
```
Notes: 
- We may want to use a skewed time scheduling, e.g. beta distribution to bias towards 1 and also avoid 0.
    - Uses Beta(α=1.5, β=1.0) distribution, then scales to [0.001, 1.0] to avoid exact 0 and 1

At inference stage, we predict the flow velocity and adjust the samples:
```
# Flow Matching Inference (Sampling)
# ==================================
# Goal: Generate samples by integrating the learned velocity field from noise to data
# 
# Notation:
#   x₁ ~ N(0, I)        : initial noise sample
#   t ∈ [1, 0]          : reverse time (1 = noise → 0 = data)
#   v_θ(x_t, t, c)      : learned velocity field
#   dx/dt = v_θ(x_t, t) : ODE to integrate
#   Δt                  : step size (negative, going backwards in time)
#   N                   : number of integration steps
#

x = torch.randn(shape)                    # x₁ ~ N(0, I): start from pure noise
num_steps = 50                            # N: number of denoising steps
dt = -1.0 / num_steps                     # Δt = -1/N: negative step (going from t=1 to t=0)

for i in range(num_steps):                # Integrate ODE from t=1 to t=0
    t = 1.0 - i / num_steps               # t_i = 1 - i/N: current time
    v = model(x, t, conditioned_inputs...)  # v_θ(x_t, t, c): predict velocity at current state
    x = x + dt * v                        # x_{t+Δt} = x_t + Δt·v_θ(x_t, t): Euler integration step
                                          # (dt is negative, so we move backwards in time)
# x is now at t=0, representing a sample from p_data
return x                                  # x₀: generated sample from the data distribution
```

## Attention 2D masks

- Row i = Query (token trying to attend)
- Column j = Key (token being attended to)
- att_2d_masks[i, j] = True means token i can attend to token j

```
# att_2d_masks[i, j] = left[0, 0, j] <= right[0, i, 0]
#                    = cumsum[j] <= cumsum[i]
```

```
import torch
mask = torch.tensor([[0, 0, 1, 1, 0, 1]])
cumsum = torch.cumsum(mask, dim=-1)
print(cumsum)
N = cumsum.shape[-1]

left = cumsum[:, None, :]       # Shape: [1, 1, N]
print("left.shape: ", left.shape)
# Right side: cumsum[:, :, None]
right = cumsum[:, :, None]      # Shape: [1, N, 1]
print("right.shape: ", right.shape)
att_2d_masks = left <= right
```

## SmolVLA attention masks

We have image observation patch tokens, language tokens, state tokens, action tokens
```
Sequence: [img₁, img₂, lang, state, act₁, act₂]
att_mask: [ 0,   0,    0,    1,    1,    1  ]
cumsum:   [ 0,   0,    0,    1,    2,    3  ]

Attention Matrix (1 = can attend, 0 = cannot):
           img₁  img₂  lang  state  act₁  act₂
    img₁ [  1     1     1     0      0     0  ]  ← bidirectional within prefix
    img₂ [  1     1     1     0      0     0  ]
    lang [  1     1     1     0      0     0  ]
    state[  1     1     1     1      0     0  ]  ← attends to all prefix + self
    act₁ [  1     1     1     1      1     0  ]  ← causal within suffix
    act₂ [  1     1     1     1      1     1  ]
```

## SmolVLA token embeddings

- Transformer contains L layers. Each layer contains H heads, and hidden size D. 
    - K, V, Q projection contains $3D^2$ parameters
    - Output projection contains $D^2$ parameters
    - MLP contains Dx4D + 4DxD = $8D^2$
    - Total we have $12LD^2$ parameters

- For language and image, it uses VLM tokenizer. 
    - It does padding for language tokens. So we need make sure we don't attend these tokens. 
    - Image and language embeddings are normalized by multiplying by sqrt(embedding_dim)

- For state, it uses linear projector

- For actions, it uses a linear projector + MLP with time conditioning
    - Actions are projected via `action_in_proj`: actions (batch, chunk_size, action_dim) → action_emb (batch, chunk_size, expert_hidden_size)
    - Time is embedded using sinusoidal_pos_embedding
        - It does an exponential spacing `period = min_period * exp(fraction * log(max_period / min_period))`
        - PE(t, 2i)   = sin(t / period_i * 2π)
        - PE(t, 2i+1) = cos(t / period_i * 2π)
    - Action and time embeddings are concatenated: [action_emb, time_emb]
    - The concatenated embedding goes through a 2-layer MLP:
        - `action_time_mlp_in`: (batch, 2*expert_hidden_size, expert_hidden_size) → (batch, expert_hidden_size, expert_hidden_size)
        - SiLU activation
        - `action_time_mlp_out`: (batch, expert_hidden_size, expert_hidden_size) → (batch, expert_hidden_size, expert_hidden_size)
    - This produces the final action token embeddings that enter the transformer

- Language, image, and state are feed into VLM model, while actions are feed into action LM expert.
    - Image is resized to (512, 512) and then break into (8, 8) patches. So the number of image token is 64. The embedding dimension is 960.
    - The number of language token could be as large as `tokenizer_max_length`. In this Push-T data, it is always 14 ('Push the T-shaped block onto the T-shaped target.'). Padding is to pad to the longest which is also 14. The embedding dimension is 960.
    - The number of state tokens is just 1. We use a linear projection to project from `max_state_dim` to the embedding dimension 960.
    - The number of action token is `chunk_size` 50, and the embedding dimension is 720 = 960 * `expert_width_multiplier`. The action in push-T is just a 2 dimensional vector,
    it is padded to `max_action_dim`
    - In summary, we have 64 + 14 + 1 = 79 tokens for VLM model and we have 50 tokens for LM action expert model. So there are 129 tokens for the input. We will also need
    pad_mask and att_mask for each token because we want to know attention relations among these tokens. 

## SmolVLMWithExpertModel

- It contains two models
    - A VLM model `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`. It uses the text_model's first num_vlm_layers layers. The vision model is used to encoder image to get embeddings. 
    - A action expert model (LM model). It is also an transformer model. It by default has the same number of layers as VLM model but will have smaller embedding size scaled by `expert_width_multiplier`. 

- Its transformer can have different number of `num_attention_heads` and `num_key_value_heads`. So basically,
multiple query heads can share the same key_value head.  

- It contains two attentions modes
    - For cross-attention mode, LM expert takes in the key/value pairs from VLM model. 
    - It just does self-attention every `self_attn_every_n_layers` layer.

- It outputs embeddings and kv caches. 
    - For self-attention module, it uses 15 heads for query, 5 heads for key and value, and each embedding dimension is 64. So 15 * 64 = 960 for prefix tokens. Each head is a 960 to 64 projection.
        - For kv values, it is a (B, L, H, D)  with L = 79 + 50 sequence length, H=15 number of heads, D=64 head dim. 
            - 79 is for prefix tokens, e.g. image, language and states. 50 is for the suffix tokens, e.g. actions. These tokens are feed into different models. One is LVM and 
            another is LM Action expert for Key, Query, Value projection. But finally, they are stacked together for attention with some causal masks.
        - It applies RoPE for the key and query.
    - For cross-attention module, it generate two attention outputs. So the LVM does self-attention on its own prefix tokens input. And the LM expert also does self-attention on its own suffix tokens input. 


## SmolVLA Select Action

- SmolVLA can predict `n_chunk_size` actions, but will execute first `n_action_steps` actions among them before next prediction.
- It needs to sample the actions based on flow matching model.
- It leverages kv caching. This is because for flow matching, only LM action expert's input token, aka, action tokens are changing. 
For VLM model, its input tokens don't change so we can cache its key & value pairs for cross-attention with LM action expert.


```
python src/lerobot/scripts/lerobot_train.py \
  --policy.path=annarborace01/smolvla-test \
  --dataset.repo_id=lerobot/pusht \
  --batch_size=64 \
  --steps=20000 \
  --output_dir=outputs/train/my_smolvla_pusht \
  --job_name=my_smolvla_pusht_training \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.push_to_hub=false
```