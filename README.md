# QLoRA_Persona
Supervised fine-tuning using QLoRA, PyTorch, Parameter-Efficient Fine-Tuning, and Transformer Reinforcement Learning.
The goal is to adapt the base LLM to a highly specific tone and persona—in this case, an **empathetic and professional customer service representative**—while minimizing VRAM usage by loading the base model in **4-bit precision**.

---

## ⚙️ Configuration

All critical hyper-parameters, model names, and the instruction system prompt are managed via the `config.yaml` file, allowing easy modification without touching the core Python script.

| Parameter | Configuration Section | Description |
| :--- | :--- | :--- |
| `base_model` | `model` | The Hugging Face ID of the model to fine-tune (e.g., `meta-llama/Meta-Llama-3-8B-Instruct`). |
| `r` & `alpha` | `lora` | Core LoRA parameters that control the **rank** and **scaling factor** of the adapter weights. |
| `system_prompt` | `training` | The custom instruction that defines the desired persona and behavior of the fine-tuned model. |
| `num_epochs` & `learning_rate` | `training` | Standard training hyper-parameters. |

---

## 🚀 Training Pipeline Overview

### Data Preparation

The script ensures that the instruction/response pairs in your dataset are formatted correctly using the **Llama 3 chat template** (`<|begin_of_text|>`, `<|start_header_id|>`, etc.) and includes the custom `system_prompt` to embed the desired customer service persona.

### Model Setup (QLoRA)

The base Llama 3 model is loaded with a `BitsAndBytesConfig` that enables **4-bit quantization**, significantly reducing memory consumption. **LoRA** is applied to the key attention and feed-forward layers (`q_proj`, `v_proj`, `o_proj`, etc.) to make the training process parameter-efficient.

### Training Execution

The `trl.SFTTrainer` handles the training loop, optimizing **only the small LoRA adapter weights** based on the provided customer service interactions. The final output is a small set of adapter files that can be merged with the original Llama 3 model for deployment.

---

## 🛠️ Set Up the Environment

### 1. Install required packages

It is highly recommended to use a virtual environment. Install PyTorch first (ensure you install the CUDA version if you plan to use a GPU).

```bash
# Install PyTorch (example for CUDA 12.1)
# pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# Install other packages
pip install transformers accelerate bitsandbytes peft trl datasets pyyaml
