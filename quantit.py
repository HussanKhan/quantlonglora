from optimum.gptq import GPTQQuantizer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
import sys
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--model", help="model name")

args = parser.parse_args()

# Ensure your script has the required modules
if not all([module in sys.modules for module in ['optimum', 'torch', 'transformers']]):
    raise ImportError("You must have optimum, torch, and transformers installed!")

# Dynamically determine the save folder based on the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
save_folder = os.path.join(script_dir, "local")
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

print("Initializing GPTQ Quantizer...")
# Dataset id from Hugging Face
dataset_id = "c4"

# GPTQ quantizer
quantizer = GPTQQuantizer(bits=4, dataset=dataset_id, model_seqlen=32768)
quantizer.quant_method = "gptq"
print("GPTQ Quantizer initialized!")

print("Loading Pre-trained Model and Tokenizer from Hugging Face...")
# Hugging Face model id
model_id = args.model
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)  # bug with fast tokenizer
model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', max_memory={0: "77GB"}, low_cpu_mem_usage=True, torch_dtype=torch.float16)
print("Pre-trained Model and Tokenizer loaded!")

print("Quantizing the Model...")
# quantize the model
quantized_model = quantizer.quantize_model(model, tokenizer)
print("Model Quantized!")

print("Saving Quantized Model to Disk...")
# save the quantized model to disk
quantized_model.save_pretrained(save_folder, safe_serialization=True)
print(f"Quantized Model saved at {save_folder}!")

print("Loading and Saving Fresh, Fast Tokenizer...")
# load fresh, fast tokenizer and save it to disk
tokenizer = AutoTokenizer.from_pretrained(model_id).save_pretrained(save_folder)
print("Fast Tokenizer saved!")

print("Saving Quantize Configuration for TGI...")
# save quantize_config.json for TGI
with open(os.path.join(save_folder, "quantize_config.json"), "w", encoding="utf-8") as f:
    quantizer.disable_exllama = False
    json.dump(quantizer.to_dict(), f, indent=2)
print("Quantize Configuration saved!")

print("Updating and Saving config.json...")
with open(os.path.join(save_folder, "config.json"), "r", encoding="utf-8") as f:
    config = json.load(f)
config["quantization_config"]["disable_exllama"] = False
with open(os.path.join(save_folder, "config.json"), "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)
print("config.json updated and saved!")

print("All operations completed successfully!")
