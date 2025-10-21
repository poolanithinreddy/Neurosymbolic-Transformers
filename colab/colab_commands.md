# NST Colab Quickstart

```bash
# 1) GPU runtime
!nvidia-smi || echo "No GPU"

# 2) Get code (GitHub route)
!git clone https://github.com/<your-username>/Neurosymbolic-Transformers.git nst
%cd nst

# 3) Install
!pip install -U pip wheel
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install -e ".[dev]"
!python -m spacy download en_core_web_sm

# 4) Quick smoke (t5-base LoRA)
!python training/train.py --config configs/colab_lora.yaml --task fever
!python eval/fever.py --ckpt outputs/ckpt --data data/fever.tsv --split dev --report outputs/fever_dev.json --device cuda

# 5) TruthfulQA
!python scripts/get_truthfulqa.py --out data/truthfulqa.tsv
!python eval/truthfulqa.py --ckpt outputs/ckpt --data data/truthfulqa.tsv --report outputs/tqa_dev.json --device cuda

# 6) COGS placeholder
!python scripts/get_cogs.py --outdir data/cogs
!python eval/cogs.py --ckpt outputs/ckpt --data data/cogs --split test --report outputs/cogs_test.json --device cuda
```
