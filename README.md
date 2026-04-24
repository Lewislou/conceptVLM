# ConceptVLM
This repository contains the implementation of **ConceptVLM**, a medical vision-language model that leverages key concept supervision to improve clinical reasoning accuracy. The method is based on the paper *"Key Concept Learning for Medical Vision Language Model with Reasoning Capabilities"*.


## 🛠️ Installation

Ensure you have Python 3.10+ and PyTorch installed. 

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers deepspeed peft accelerate
```

Install project dependencies, please refer to [InternVL](https://github.com/OpenGVLab/InternVL): 
```bash
pip install -e .
```
## 🔧 Usage

### 1. Train the Model

Run training using provided shell scripts:

```bash
# Example: Stage 3 training with 34B model
bash shell/train_stage3.sh zero_stage3_config_34b.json
```

### 2. Evaluate the Model

```bash
bash evaluate.sh path/to/checkpoint
```

## Citation

If you use this code or methodology in your research, please cite our paper:

```bibtex

```

---

## Contact

For questions or collaboration, please contact: louwei@zjnu.edu.cn

---
