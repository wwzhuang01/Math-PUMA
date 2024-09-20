import os
import fire
import json
import torch
import pickle
import regex as re
from PIL import Image
from tqdm import tqdm
from loguru import logger
from typing import List, Dict, Union
from transformers.processing_utils import ProcessorMixin
from models.qwen2 import Qwen2vlmProcessor, Qwen2vlmForConditionalGeneration

USER_HINT = """
Hint: Please answer the question and provide the final answer at the end.\nQuestion: {question}
""".strip()


def extract_answer_try_all_methods(s, remove_parentheses=True):
    s = s.replace('[UNUSED_TOKEN_145]', '')
    if not s:
        return s
    last_plus_index = s.rfind("†")
    if last_plus_index != -1:  # "+" found in the string
        return s[last_plus_index+1:]

    matches = re.findall(
        r"(answer:|Answer:|therefore,|so,|Therefore,|So,|Thus,|thus,)([\s\S]*?)(?=answer:|Answer:|therefore,|so,|Therefore,|So,|Thus,|thus,|$)", s)
    if matches:
        ans = matches[-1][1].strip()
        return ans.strip()

    return s.strip()


def single_inference(
    processor: ProcessorMixin,
    model: Qwen2vlmForConditionalGeneration,
    cuda_device: int,
    input_prompt: str = '',
    image: Union[str, Image.Image] = None,
    system_prompt: str = '',
    dtype: torch.dtype = torch.bfloat16,
    **kwargs
) -> str:
    conv = [{"role": "system", "content": [
        {"type": "text", "text": system_prompt}]}] if system_prompt else []
    content = [
        {"type": "text", "text": input_prompt}
    ]
    if image:
        content.append({"type": "image"})
    conv.append({
        "role": "user",
        "content": content
    })

    prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
    if isinstance(image, str):
        raw_image = Image.open(image).convert('RGB')
    elif isinstance(image, Image.Image):
        raw_image = image.convert('RGB')
    else:
        raw_image = None
    inputs = processor(prompt, raw_image, return_tensors='pt').to(
        cuda_device, torch.bfloat16)

    generation_config = dict(
        max_new_tokens=kwargs.pop('max_new_tokens', 768),
        do_sample=kwargs.pop('do_sample', True),
        temperature=kwargs.pop('temperature', 0.1),
        top_p=kwargs.pop('top_p', 0.8),
        pad_token_id=kwargs.pop(
            'pad_token_id', processor.tokenizer.eos_token_id),
        repetition_penalty=kwargs.pop('repetition_penalty', 1.0)
    )
    outputs = model.generate(**inputs, **generation_config)
    response = processor.decode(outputs[0], skip_special_tokens=False)

    if '<|im_start|>assistant' in response:     # qwen2vlm
        response = response.split(
            '<|im_start|>assistant')[-1].replace(input_prompt, '').replace('<|im_end|>', '').strip()
    elif '[/INST]' in response:                 # mathstralvlm
        response = response.split(
            '[/INST]')[-1].replace(input_prompt, '').strip()

    return response


def inference(
    start_index: int,
    end_index: int,
    cuda_device: int,
    dataset_name: str = 'mathverse',
    data_path: str = './data/mathverse/testmini.json',
    model_path: str = '<mathpuma>',
    model_type: str = 'qwen2vlm',
    dtype: torch.dtype = torch.bfloat16,
    *args,
    **kwargs
):
    if all(name not in model_path.lower() for name in ['qwen', 'mathstral']):
        logger.warning(
            f"supported model_name not in {model_path}, make sure you have selected the correct vlm model")

    logger.info('Loading data...')
    data = []
    if data_path.endswith('.json'):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif data_path.endswith('.jsonl'):
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                data.append(json.loads(line.strip()))
    elif any(data_path.endswith(suffix) for suffix in ['pk', 'pkl']):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

    if dataset_name == 'mathvista':
        data = [data[f"{i}"] for i in range(start_index + 1, end_index + 1)]
    else:
        data = data[start_index:end_index]

    logger.info('Loading model...')

    model_class = Qwen2vlmForConditionalGeneration
    processor_class = Qwen2vlmProcessor

    model = model_class.from_pretrained(
        model_path,
        torch_dtype=dtype,
    ).to(cuda_device)
    processor = processor_class.from_pretrained(model_path)

    if dataset_name == 'mathverse':
        inference_mathverse(processor, model, cuda_device,
                            data, *args, **kwargs)
    elif dataset_name == 'mathvista':
        inference_mathvista(processor, model, cuda_device,
                            data, *args, **kwargs)
    elif dataset_name == 'mathvision':
        inference_mathvision(processor, model, cuda_device,
                             data, *args, **kwargs)
    elif dataset_name == 'wemath' or dataset_name == 'we-math':
        inference_wemath(processor, model, cuda_device, data, *args, **kwargs)
    elif dataset_name == 'geoqa':
        inference_geoqa(processor, model, cuda_device, data, *args, **kwargs)
    else:
        raise NotImplementedError(f"Not supported for dataset: {dataset_name}")


def inference_mathverse(
    processor: ProcessorMixin,
    model: Union[Qwen2vlmForConditionalGeneration, MathstralvlmForConditionalGeneration],
    cuda_device: int,
    data: List[Dict],
    image_root: str = './datasets/MathVerse',
    output_path: str = './outputs/mathverse_output.json',
    *args,
    **kwargs
):
    for i, d in tqdm(enumerate(data), total=len(data), desc=f'[Device: {cuda_device}]'):
        prompt = d['query_cot']
        if d['image']:
            image_path = os.path.join(image_root, d['image'])
        else:
            image_path = None

        response = single_inference(
            processor, model, cuda_device, input_prompt=prompt, image=image_path)
        extraction = extract_answer_try_all_methods(response)

        new_item = {
            **d,
            'model_answer': response,
            'extraction': extraction,
        }
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(new_item, ensure_ascii=False) + '\n')


def inference_mathvista(
    processor: ProcessorMixin,
    model: Qwen2vlmForConditionalGeneration,
    cuda_device: int,
    data: List[Dict],
    image_root: str = './datasets/MathVerse/images',
    output_path: str = '../temp.jsonl',
    *args,
    **kwargs
):
    for i, d in tqdm(enumerate(data), total=len(data), desc=f'[Device: {cuda_device}]'):
        question = data[i]['question']
        image_path = os.path.join(image_root, data[i]['image'])
        prompt = USER_HINT.format(question=question)

        response = single_inference(
            processor, model, cuda_device, input_prompt=prompt, image=image_path)
        extraction = extract_answer_try_all_methods(response)

        new_item = {
            **d,
            'query': prompt,
            'response': response,
            'extraction': extraction,
        }
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(new_item, ensure_ascii=False) + '\n')


def inference_mathvision(*args, **kwargs):
    raise NotImplementedError("Not supported for mathvision yet.")


def inference_wemath(
    processor: ProcessorMixin,
    model: Qwen2vlmForConditionalGeneration,
    cuda_device: int,
    data: List[Dict],
    image_root: str = './datasets/MathVerse/images',
    output_path: str = '../temp.jsonl',
    *args,
    **kwargs
):
    base_prompt = """Now, we require you to solve a multiple-choice math question. We will provide
    you with the relevant knowledge concepts of this question for your reference.
    Please briefly describe your thought process and provide the final answer(option).
    Knowledge concept: {knowledge_concept}
    Question: {question}
    Option: {option}
    Regarding the format, please answer following the template below, and be
    sure to include two <> symbols:
    <Thought process>: <<your thought process>> <Answer>: <<your option>>""".replace('\n', ' ')

    for i, d in tqdm(enumerate(data), total=len(data), desc=f'[Device: {cuda_device}]'):
        question = d['question']
        option = d['option']
        knowledge_concept = d['knowledge concept description']
        image_path = os.path.join(image_root, d['image_path'])
        prompt = base_prompt.format(
            question=question, option=option, knowledge_concept=knowledge_concept)

        response = single_inference(
            processor, model, cuda_device, input_prompt=prompt, image=image_path)
        extraction = extract_answer_try_all_methods(response)

        new_item = {
            **d,
            'response': response,
            'extraction': extraction,
        }
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(new_item, ensure_ascii=False) + '\n')


def inference_geoqa(
    processor: ProcessorMixin,
    model: Qwen2vlmForConditionalGeneration,
    cuda_device: int,
    data: List[Dict],
    output_path: str = '../temp.jsonl',
    *args, **kwargs
):
    label2id = {
        0: 'A',
        1: 'B',
        2: 'C',
        3: 'D',
    }

    for i, item in tqdm(enumerate(data), total=len(data), desc=f'[Device: {cuda_device}]'):
        image = Image.fromarray(item['image']).convert('RGB')
        choices = [f'{c}. {choice}' for c,
                   choice in zip('ABCD', item['choices'])]
        question = item['subject'] + '\n' + ' '.join(choices)
        answer = label2id[item['label']]

        response = single_inference(
            processor, model, cuda_device, input_prompt=question, image=image)
        extraction = extract_answer_try_all_methods(response)

        new_item = {
            'question': question,
            'answer': answer,
            'model_answer': response,
            'extraction': extraction,
        }
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(new_item, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    fire.Fire(inference)
