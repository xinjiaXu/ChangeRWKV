# ChangeRWKV server setup

The data loader reads `/home/xxj/Data` directly. It does not copy or symlink dataset files. Checkpoints, logs, and predictions are stored under `ChangeRWKV_work_dirs/<dataset>/`.

## Environment and CUDA extension

From `/home/xxj/code/DINOCLIP_comparative_test/ChangeRWKV`:

```bash
bash setup_server_env.sh
conda activate changerwkv-rs
```

The upstream README specifies Python 3.10, PyTorch 2.5.1/CUDA 12.1 as its example, and `ninja`. This server uses RTX 5090, so the isolated environment uses PyTorch 2.7.1/CUDA 12.8. ChangeRWKV JIT-compiles `model/cuda/wkv_cuda.cu` when the model is imported; the setup script also installs the CUDA 12.8 toolkit into this Conda environment. An installed `g++` is required. The first train/test process compiles the extension, so allow extra startup time.

The NVIDIA driver reported by `nvidia-smi` is not the CUDA compiler. The setup script verifies `nvcc` and PyTorch's CUDA/GPU detection after installation. NVIDIA documents Conda installation of CUDA Toolkit components; the environment script pins the CUDA 12.8 toolkit label to match the PyTorch wheel.

## Pretrained encoder

The official README provides the Hierarchical RWKV encoder checkpoint URL. Download it into the expected path:

```bash
mkdir -p ckpt
wget -O ckpt/vrwkv_b.pth https://github.com/ChangeRWKV/ChangeRWKV/releases/download/v0.1/vrwkv_b.pth
```

Training checks that this file exists before starting. To use a different path, set `PRETRAINED=/path/to/vrwkv_b.pth`.

## Data paths

Optionally check all expected split folders:

```bash
bash prepare_server_data.sh /home/xxj/Data
```

The loader accepts the upstream flat layout with `list/*.txt` and the server's split layout. For split layout it enumerates filenames from each split's label folder, so no list files are needed. Matching filenames must exist in T1, T2, and label folders.

| Key | Dataset root | T1 | T2 | Mask |
| --- | --- | --- | --- | --- |
| `levircd` | `/home/xxj/Data/LEVIR_CD` | `A` | `B` | `label` |
| `levircdplus` | `/home/xxj/Data/LEVIR_CD_PLUS/cut_data` | `A` | `B` | `label` |
| `cdd` | `/home/xxj/Data/CDD` | `A` | `B` | `OUT` |
| `whucd` | `/home/xxj/Data/WHUCD` | `A` | `B` | `label` |
| `sysucd` | `/home/xxj/Data/SYSU_CD` | `time1` | `time2` | `label` |

## Train

```bash
bash run_train.sh levircd
bash run_train.sh levircdplus
bash run_train.sh cdd
bash run_train.sh whucd
bash run_train.sh sysucd
```

The wrappers keep upstream defaults: base model, image size 256, batch size 32, 200 epochs, Adam, learning rate `1e-5`, cosine schedule, 20 warmup epochs, and weight decay `1e-4`. WHU validation uses batch size 4 in the original code. Runtime defaults are one visible GPU and 8 data workers; use `GPU_IDS=1` to select GPU 1, or `GPU_IDS=0,1 DEVICES=2` to use both GPUs.

Each epoch writes `last.pth`; `EpochN.pth` files are also saved from epoch 30 onward. The code logs validation metrics but does not select a best checkpoint. Checkpoints, `config.json`, `res.out`, and XLSX logs are written below `ChangeRWKV_work_dirs/<key>/`.

## Test

```bash
bash run_test.sh levircd ChangeRWKV_work_dirs/levircd/LEVIR/ChangeRWKV-b/last.pth
```

Choose another dataset key and checkpoint as needed. Test uses batch size 1 and writes predicted masks to a timestamped directory under that dataset's work directory, so previous prediction outputs are not replaced.
