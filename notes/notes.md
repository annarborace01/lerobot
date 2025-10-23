## Notes:
- For RTX6000 blackwell, it requires torch at least 2.71 + cu128
```
pip install torch==2.7.1 torchvision==0.22.1 \
-extra-index-url https://download.pytorch.org/whl/cu128 
```

- torchcodex seems not working for ffmpeg 6 somehow
```
conda install ffmpeg=7.1.1 -c conda-forge
```

- 