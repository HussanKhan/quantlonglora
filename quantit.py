from optimum.gptq import GPTQQuantizer


# Dataset id from Hugging Face
dataset_id = "wikitext2"

# GPTQ quantizer
quantizer = GPTQQuantizer(bits=4, dataset=dataset_id, model_seqlen=4096)
quantizer.quant_method = "gptq"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Hugging Face model id
model_id = "expansezero/llama232klonglora"

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False) # bug with fast tokenizer
model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', max_memory={0: "77GB", 1: "77GB"}, low_cpu_mem_usage=True, torch_dtype=torch.float16) # we load the model in fp16 on purpose

import os
import json

# quantize the model
quantized_model = quantizer.quantize_model(model, tokenizer)

# save the quantize model to disk
save_folder = "./local"
quantized_model.save_pretrained(save_folder, safe_serialization=True)

# load fresh, fast tokenizer and save it to disk
tokenizer = AutoTokenizer.from_pretrained(model_id).save_pretrained(save_folder)

# save quantize_config.json for TGI
with open(os.path.join(save_folder, "quantize_config.json"), "w", encoding="utf-8") as f:
  quantizer.disable_exllama = False
  json.dump(quantizer.to_dict(), f, indent=2)

with open(os.path.join(save_folder, "config.json"), "r", encoding="utf-8") as f:
  config = json.load(f)
  config["quantization_config"]["disable_exllama"] = False
  with open(os.path.join(save_folder, "config.json"), "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)
