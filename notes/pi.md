- I think we need correct version of transformers from HF to work with pi `transformers.models.siglip`
```
pip install "lerobot[pi]"
```

- You need sign the agreement https://huggingface.co/google/paligemma-3b-pt-224 to be able to access google models for pi.

- It requires `src/lerobot/datasets/v30/augment_dataset_quantile_stats.py` to compute quantile properties.

- `max_action_dim` by default is 32. So the action will be padded to 32. Then `nn.Linear(config.max_action_dim, action_expert_config.width)` is used to project the action into embedding space.

```
python src/lerobot/scripts/lerobot_train.py \
  --policy.path=lerobot/pi05_base \
  --dataset.repo_id=lerobot/pi05_base \
  --dataset.repo_id=lerobot/pusht \
  --batch_size=16 \
  --steps=20000 \
  --output_dir=outputs/train/my_pi05_pusht \
  --job_name=my_pi05_pusht_training \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.push_to_hub=false
```
