# Deprecated-API OOD Region — One-Class SVM

## 1. Set Up the Environment

```bash
conda create -n ocsvm python=3.10
conda activate ocsvm
pip install -r requirements.txt
```

## 2. Run

```bash
sudo nohup bash train_ood.sh > logs/train_ood.log 2>&1 &
```
