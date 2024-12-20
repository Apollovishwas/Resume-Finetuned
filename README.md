# Mistral-7B Resume Fine-tuning Project
This project demonstrates how to fine-tune the Mistral-7B-Instruct model for resume-based question answering using LoRA (Low-Rank Adaptation) technique.
## Requirements
bash
pip install -q -U transformers datasets accelerate peft trl bitsandbytes wandb sentencepiece
## Project Structure
The project uses the following key components:
Base Model: Mistral-7B-Instruct-v0.1
Training Method: LoRA (Low-Rank Adaptation)
Quantization: 4-bit quantization using BitsAndBytes
## Configuration
### Model Settings:
Load in 4-bit precision
Quantization type: nf4
Compute dtype: float16
Double quantization enabled
### LoRA Parameters:
Rank (r): 16
Alpha: 32
Dropout: 0.05
Target modules: up_proj, down_proj, gate_proj, k_proj, q_proj, v_proj, o_proj
## Training Parameters
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
## Dataset Preparation
The training data should be in CSV format with 'question' and 'answer' columns. The script preprocesses the data into the following format:

[INST] Question [/INST]
Answer

## Usage
Set up your Hugging Face token
Prepare your dataset in CSV format
Run the training script
The model will be saved and can be pushed to Hugging Face Hub

## Example usage:
```
prompt = "What is your question?"
instruction = f"[INST] {prompt} [/INST]"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=128)
result = pipe(instruction)
```
## Future Improvements
Consider using Axolotl for better fine-tuning
Implement evaluation using LM Evaluation Harness
Explore alternative quantization methods (GPTQ, GGUF, ExLlamav2, AWQ)
Test with different model architectures
