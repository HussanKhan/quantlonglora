from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import logging
from datasets import load_dataset
import pandas as pd
import os
import torch

def format_prompt(instruction, output):
    return f"""Below is an instruction that describes a task. 
Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}
"""

data = load_dataset('Yukang/LongAlpaca-12k', split='train')
df = pd.concat([pd.DataFrame(data['instruction'], columns=['instruction']), pd.DataFrame(data['output'], columns=['output'])], axis=1)

df['instruction_len'] = df['instruction'].apply(lambda x: len(x))
df['output_len'] = df['output'].apply(lambda x: len(x))

df['prompt'] = df.apply(lambda x: format_prompt(x['instruction'], x['output']), axis=1)

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)

pretrained_model_dir = "expansezero/llama232klonglora"
quantized_model_dir = "local"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
df.sort_values(by='output_len', inplace=True, ascending=False)
print(df)
print('--------------------------------')
examples = [tokenizer(e) for e in df['prompt'].tolist()[:256]]
print('Tokenizing finished')

quantize_config = BaseQuantizeConfig(
    bits=4,  # quantize model to 4-bit
    group_size=128,  # it is recommended to set the value to 128
    desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
)

# load un-quantized model, by default, the model will always be loaded into CPU memory
model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)
print('Model loaded into CPU memory')

# quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
model.quantize(examples)

# save quantized model
model.save_quantized(quantized_model_dir)

# save quantized model using safetensors
model.save_quantized(quantized_model_dir, use_safetensors=True)
