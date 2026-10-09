# Theft Watch: dev machine setup (edge emulation + training)

Replicates the Phase 0–1 setup on any Linux PC with an NVIDIA GPU (Turing or newer).
Verified first on 2026-10-06: Dell Inspiron 3910, NVIDIA T1000 4 GB, driver 610.57.

## What you get

| Container | Image | Purpose |
| --- | --- | --- |
| `edge` (tw-edge) | `nvcr.io/nvidia/deepstream:9.1-triton-multiarch` | The "emulated Jetson": DeepStream 9.1, CUDA 13.2, TensorRT 10.16, GStreamer. Same release exists for Jetson Orin/Thor (JetPack 7.2). |
| `trainer` (tw-trainer) | built from `training/Dockerfile` | PyTorch (official CUDA 12.8 wheels), video tools, transformers, YOLO, ONNX, MLflow, Jupyter |

The project folder is mounted at `/workspace` in both containers.

## Requirements

- Ubuntu 22.04 or 24.04 (x86_64)
- NVIDIA GPU, Turing or newer (RTX 20xx/30xx/40xx/50xx, T-series, A-series)
- NVIDIA driver **595.58 or newer** (DeepStream 9.1 requirement)
- ~150 GB free disk (edge image is ~15–20 GB; videos grow fast)
- Free NVIDIA NGC account (to pull DeepStream)

## Phase 0: prepare the machine

```bash
# 0.1 Check the system
lsb_release -a; df -h ~; free -h
nvidia-smi                       # driver >= 595.58, note GPU name + memory

# 0.2 Docker (official apt repo; remove the snap version first if present: sudo snap remove docker)
sudo apt-get update && sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" | sudo tee /etc/apt/sources.list.d/docker.list
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER    # then log out and back in

# 0.3 NVIDIA Container Toolkit (lets containers use the GPU)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 0.4 Test: a container must show the GPU
docker run --rm --gpus all ubuntu nvidia-smi

# 0.5 NGC login (ngc.nvidia.com -> Setup -> Generate API Key). Username is literally $oauthtoken
docker login nvcr.io -u '$oauthtoken'
```

## Phase 1: containers and checks

```bash
# 1.1 Get the project
git clone <your-repo-url> ~/DevProjects/theft-watch     # or copy the folder
cd ~/DevProjects/theft-watch

# 1.2 Pull / build
docker compose pull edge          # large download
docker compose build trainer      # 10–20 min

# 1.3 Start
docker compose up -d edge trainer

# 1.4 Check training container: GPU, memory, kernels for this GPU, quick speed test
docker compose exec trainer python training/check_gpu.py

# 1.5 Check emulated edge: plugins, hardware decode, 2 streams through a detector
#     First run on a new GPU builds a TensorRT engine (a few minutes)
docker compose exec edge bash edge/check_deepstream.sh
```

Pass criteria: `check_gpu.py` ends with `OK`; `check_deepstream.sh` shows `decode: OK`,
`batches rendered ... average: N` with N well above 30, `dropped: 0`, and exit code 0.

## Reference results

| Date | Machine | GPU | Detector test (2 x 1080p H.264, sample ResNet-18, FP16, batch 2) |
| --- | --- | --- | --- |
| 2026-10-06 | Inspiron 3910 | T1000 4 GB, driver 610.57 | ~105 fps per stream, 0 dropped |

Add a row for every new machine, so results stay comparable.

## Notes and known issues

- **TensorRT engines are per GPU.** Never copy `.engine` files between machines; they are rebuilt on first run.
- **Harmless warning:** `Deserialize engine failed ... _b30_gpu0_fp16.engine open error` in the edge check. The sample config looks for a 30-stream engine first, then uses/builds the 2-stream one.
- **NGC PyTorch images do not support Turing** (T1000, RTX 20xx) in recent releases; that is why the trainer uses official PyTorch wheels. On Ampere or newer you may switch to an NGC image if preferred.
- **Other PyTorch CUDA build:** `docker compose build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu130 trainer`
- **Don't run `deepstream-app` sample configs headless with edited sinks**; it hung in our test. The check uses `gst-launch-1.0` with the same plugins instead.
- **Ultralytics YOLO is AGPL-3.0**: fine for learning; a closed commercial product needs a paid licence or an Apache-licensed detector.
- **With a bigger GPU (12 GB+):** behaviour-model option B (fine-tune a small video model) becomes practical locally; 24 GB+ allows option C (full VideoMAE fine-tune).
