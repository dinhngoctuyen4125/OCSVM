# Deprecated-API OOD Region — One-Class SVM

## 1. Set Up the Environment

```bash
conda create -n ocsvm python=3.10
conda activate ocsvm
pip install -r requirements.txt
```

`requirements.txt` chỉ để `torch>=2.2`. Nếu cần đúng build của server tham chiếu
(nightly CUDA 12.8, không có trên PyPI), cài riêng trước:

```bash
pip install --pre torch==2.12.0.dev20260408+cu128 \
    --index-url https://download.pytorch.org/whl/nightly/cu128
```

## 2. Run

```bash
sudo nohup bash train_ood.sh > logs/train_ood.log 2>&1 &
```
