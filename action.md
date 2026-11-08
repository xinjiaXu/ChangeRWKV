# ChangeRWKV Server Actions

Server checkout: `/home/xxj/code/DINOCLIP_comparative_test/ChangeRWKV`  
Dataset root: `/home/xxj/Data`

## Pull reproduce branch

First time:

```bash
cd /home/xxj/code/DINOCLIP_comparative_test/ChangeRWKV
git fetch origin
git switch --track -c reproduce origin/reproduce
```

Later updates:

```bash
cd /home/xxj/code/DINOCLIP_comparative_test/ChangeRWKV
git switch reproduce
git pull --ff-only
```

## Environment, data, and optional pretrained weight

```bash
bash setup_server_env.sh
conda activate changerwkv-rs
mkdir -p ckpt
wget -O ckpt/vrwkv_b.pth https://github.com/ChangeRWKV/ChangeRWKV/releases/download/v0.1/vrwkv_b.pth
bash prepare_server_data.sh /home/xxj/Data
```

The environment includes the CUDA 12.8 toolkit and Ninja. The model JIT-compiles its WKV CUDA extension at first import, so `nvcc` and `g++` must be available. The first train or test launch spends extra time compiling.

The official Hierarchical RWKV encoder checkpoint is optional. To train from random initialization without downloading it, use `PRETRAINED=none`; this is a scratch-trained configuration and differs from the pretrained setup in the paper.

## LEVIR-CD and LEVIR-CD-PLUS image sizes

The model's fusion module is configured for 256x256 inputs. The current server datasets may contain original 1024x1024 images, while LEVIR-CD-PLUS may already contain cropped tiles. The `levircd` and `levircdplus` launchers now split images and masks into aligned, non-overlapping 256x256 patches while loading; 1024x1024 inputs become 16 samples, and already-cropped 256x256 files remain one sample. No dataset files are modified or duplicated. Any input dimensions must be divisible by 256.

## Model complexity

After activating `changerwkv-rs`, install the profiling dependency and profile one paired 256x256 input. This imports/compiles the CUDA WKV extension but does not load the pretrained checkpoint:

```bash
python -m pip install -r requirements_profile.txt
CUDA_VISIBLE_DEVICES=0 python profile_model.py --model-size base --device cuda:0 \
  --output ChangeRWKV_work_dirs/profile_256.json
```

The output lists unsupported operators. fvcore cannot count the custom WKV CUDA kernel, so its reported FLOPs exclude that work and must be treated as a partial count.

## Train

```bash
bash run_train.sh levircd
bash run_train.sh levircdplus
bash run_train.sh cdd
bash run_train.sh whucd
bash run_train.sh sysucd
```

Defaults: base model, 256 input, batch 32, 200 epochs, Adam, lr `1e-5`, cosine scheduler, 20 warmup epochs, weight decay `1e-4`. Set `BATCH_SIZE=2` (or another value) to control memory. Select GPU 1 with `GPU_IDS=1`; use both with `GPU_IDS=0,1 DEVICES=2`.

Example for the raw LEVIR-CD dataset:

```bash
PRETRAINED=none GPU_IDS=1 DEVICES=1 NUM_WORKERS=4 BATCH_SIZE=2 bash run_train.sh levircd
```

LEVIR-CD-PLUS uses the same on-the-fly patch handling:

```bash
PRETRAINED=none GPU_IDS=1 DEVICES=1 NUM_WORKERS=4 BATCH_SIZE=2 bash run_train.sh levircdplus
```

Each run directory retains `last.pth` and at most the two newest `EpochN.pth` files. Older numbered checkpoints in that run directory are pruned at startup and as training continues. The model does not choose a best-validation checkpoint.

## Test

```bash
bash run_test.sh levircd ChangeRWKV_work_dirs/levircd/LEVIR/ChangeRWKV-b/last.pth
bash run_test.sh levircdplus ChangeRWKV_work_dirs/levircdplus/LEVIR+/ChangeRWKV-b/last.pth
bash run_test.sh cdd ChangeRWKV_work_dirs/cdd/CDD/ChangeRWKV-b/last.pth
bash run_test.sh whucd ChangeRWKV_work_dirs/whucd/WHU/ChangeRWKV-b/last.pth
bash run_test.sh sysucd ChangeRWKV_work_dirs/sysucd/SYSU/ChangeRWKV-b/last.pth
```

Training writes `last.pth` every epoch and `EpochN.pth` from epoch 30 onward; it does not select a best-validation checkpoint. Test masks are written to a fresh timestamped directory under the corresponding work directory.
