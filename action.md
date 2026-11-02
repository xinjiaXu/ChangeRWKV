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

## Environment, data, and pretrained weight

```bash
bash setup_server_env.sh
conda activate changerwkv-rs
mkdir -p ckpt
wget -O ckpt/vrwkv_b.pth https://github.com/ChangeRWKV/ChangeRWKV/releases/download/v0.1/vrwkv_b.pth
bash prepare_server_data.sh /home/xxj/Data
```

The environment includes the CUDA 12.8 toolkit and Ninja. The model JIT-compiles its WKV CUDA extension at first import, so `nvcc` and `g++` must be available. The first train or test launch spends extra time compiling.

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

Defaults: base model, 256 input, batch 32, 200 epochs, Adam, lr `1e-5`, cosine scheduler, 20 warmup epochs, weight decay `1e-4`. Select GPU 1 with `GPU_IDS=1`; use both with `GPU_IDS=0,1 DEVICES=2`.

## Test

```bash
bash run_test.sh levircd ChangeRWKV_work_dirs/levircd/LEVIR/ChangeRWKV-b/last.pth
bash run_test.sh levircdplus ChangeRWKV_work_dirs/levircdplus/LEVIR+/ChangeRWKV-b/last.pth
bash run_test.sh cdd ChangeRWKV_work_dirs/cdd/CDD/ChangeRWKV-b/last.pth
bash run_test.sh whucd ChangeRWKV_work_dirs/whucd/WHU/ChangeRWKV-b/last.pth
bash run_test.sh sysucd ChangeRWKV_work_dirs/sysucd/SYSU/ChangeRWKV-b/last.pth
```

Training writes `last.pth` every epoch and `EpochN.pth` from epoch 30 onward; it does not select a best-validation checkpoint. Test masks are written to a fresh timestamped directory under the corresponding work directory.
