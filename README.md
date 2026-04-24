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
### 1. Add your datases and dictionary json file
Add your data path in shell\data\internvl_1_2_finetune_custom.json.
Add your dictionary json file and tokenizer path in internvl\model\internvl_chat\modeling_internvl_chat.py


### 2. Train the Model

Run training using provided shell scripts:

```bash
bash shell\internvl2.0\2nd_finetune\internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_lora.sh
```

### 3. Evaluate the Model

```bash
bash evaluate.sh path/to/checkpoint vqa-textvqa-val --dynamic 
```

## Citation

If you use this code or methodology in your research, please cite our paper:

```bibtex

```

---

## Contact

For questions or collaboration, please contact: louwei@zjnu.edu.cn

---
