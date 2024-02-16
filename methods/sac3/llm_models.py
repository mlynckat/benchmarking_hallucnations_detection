import os
# update your cache dir
# os.environ['HUGGINGFACE_HUB_CACHE'] = '/home/ec2-user/SageMaker/hf_cache'
# os.environ['HF_HOME'] = '/home/ec2-user/SageMaker/hf_cache'

import openai
import torch
from dotenv import load_dotenv
from peft import PeftModel
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import time


# Initialize OpenAI API
# openai.api_key = 'your openai key'
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

def call_openai_model(prompt, model, temperature):
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature
    )
    try:
        output = response.choices[0].message.content
        cost = response.usage.total_tokens
    except Exception:
        output = 'do not have reponse from chatgpt'
        cost = 0
    return output, cost


def call_guanaco_33b(prompt, max_new_tokens):
    # 16 float
    model_name = "huggyllama/llama-30b"
    adapters_name = 'timdettmers/guanaco-33b'
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        # offload_folder="/home/ec2-user/SageMaker/hf_cache",
        max_memory={i: '16384MB' for i in range(torch.cuda.device_count())},  # V100 16GB
    )
    model = PeftModel.from_pretrained(model, adapters_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # prompt
    formatted_prompt = (
        f"A chat between a curious human and an artificial intelligence assistant."
        f"The assistant gives helpful, concise, and polite answers to the user's questions.\n"
        f"### Human: {prompt} ### Assistant:"
    )
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda:0")
    outputs = model.generate(inputs=inputs.input_ids, max_new_tokens=max_new_tokens)
    res = tokenizer.decode(outputs[0], skip_special_tokens=True)
    res_sp = res.split('###')
    output = res_sp[1] + res_sp[2]

    return output, 0


def call_falcon_7b(pipeline, tokenizer, prompt, max_new_tokens):
    # 16 float

    sequences = pipeline(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    for seq in sequences:
        print(f"Sequence: {seq['generated_text']}")
        res = seq['generated_text']
    res = res.split(prompt)[-1]
    print(f"Result: {res}")

    return res, 0

def call_starling_7b(pipeline, tokenizer, prompt, max_new_tokens, temperature):
    sequences = pipeline(
        f"GPT4 Correct User: {prompt}<|end_of_turn|>GPT4 Correct Assistant: ",
        max_length=256,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=temperature,
    )
    for seq in sequences:
        print(f"Sequence: {seq['generated_text']}")
        res = seq['generated_text']
    res = res.split("GPT4 Correct Assistant: ")[-1]
    print(f"Result: {res}")

    return res, 0

def call_llama(prompt, temperature):
    client = openai.OpenAI(api_key=TOGETHER_API_KEY,
                           base_url='https://api.together.xyz',
                           )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are an AI assistant",
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="meta-llama/Llama-2-70b-chat-hf",
        max_tokens=256,
        temperature=temperature,
    )
    try:
        output = chat_completion.choices[0].message.content
        cost = chat_completion.usage.total_tokens
    except Exception:
        output = 'do not have reponse from llama'
        cost = 0

    return output, cost