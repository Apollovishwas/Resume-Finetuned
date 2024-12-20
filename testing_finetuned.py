# -*- coding: utf-8 -*-
"""Testing_Finetuned.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1AoSew8xnv5Cqgwitrncn135FhAfVYc6g

# Installing Libs
"""

!pip install transformers
!pip install torch
!pip install accelerate
!pip install bitsandbytes

"""# Download the model and Test
#### You need Huggingface Token for this

#### Replace 'model_name' with your 'new_model' from the other one
"""

!pip install gradio
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Load model with optimizations
model_name = "veechan/Mistral-7B-vishwas-resume-finetuned"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128
)

def generate_response(prompt):
    instruction = f"<s>[INST] {prompt} [/INST]"
    response = pipe(instruction)
    answer = response[0]['generated_text'][response[0]['generated_text'].find("[/INST]") + 7:].strip()
    return answer

# Create Gradio interface
demo = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(label="Enter your question about Vishwas's resume"),
    outputs=gr.Textbox(label="Response"),
    title="Vishwas Resume Chat Assistant"
)

demo.launch(share=True)
