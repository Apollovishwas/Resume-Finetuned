# Mistral-7B Resume Fine-tuning Project

This project demonstrates how to fine-tune the Mistral-7B-Instruct model for resume-based question answering using LoRA (Low-Rank Adaptation) technique.

Part 1 - Run 'Save_FineTuned_Model.ipynb' to train and save your final model to huggingface
Part 2 - Run 'Testing_Finetuned.ipynb' to load the saved model from HF and run it with Gradio UI(just an UI Library)

## Part 1 - FineTuning and Saving Model 

### Requirements
```
pip install -q -U transformers datasets accelerate peft trl bitsandbytes wandb sentencepiece
```
###  Project Structure
The project uses the following key components:
Base Model: Mistral-7B-Instruct-v0.1
Training Method: LoRA (Low-Rank Adaptation)
Quantization: 4-bit quantization using BitsAndBytes
###  Configuration
####  Model Settings:
Load in 4-bit precision
Quantization type: nf4
Compute dtype: float16
Double quantization enabled
####  LoRA Parameters:
Rank (r): 16
Alpha: 32
Dropout: 0.05
Target modules: up_proj, down_proj, gate_proj, k_proj, q_proj, v_proj, o_proj
###  Training Parameters
```
training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=10,
    gradient_accumulation_steps=1,
    evaluation_strategy="steps",
    eval_steps=1000,
    learning_rate=2e-4,
    lr_scheduler_type="linear",
    warmup_steps=10
)
```
###  Dataset Preparation
The training data should be in CSV format with 'question' and 'answer' columns. The script preprocesses the data into the following format:

[INST] Question [/INST]
Answer

### Training 

Trained using LoRA and saved to HuggingFace repo of yours


## Part 2 - Testing with Gradio UI 

### Requirements
transformers
torch
accelerate
bitsandbytes
gradio
```
pip install transformers torch accelerate bitsandbytes gradio
```

### Web Interface
The application provides a simple web interface where users can:
Input questions about Vishwas's resume
Receive AI-generated responses based on the resume content
Access the interface through a shareable link

