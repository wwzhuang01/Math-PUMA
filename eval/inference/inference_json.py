from lmdeploy import (
    pipeline,
    ChatTemplateConfig,
    TurbomindEngineConfig,
    GenerationConfig
)
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN
import os
from tqdm import tqdm
import re
import json
import math
import fire
import pickle
from PIL import Image
from typing import List, Dict


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


def inference(
    dataset_name: str = 'mathvista',
    data_path: str = './data/mathvista/testmini.json',
    model_type: str = 'deepseek-vl',
    model_path: str = './models/deepseek-math-vl-aligned',
    image_root: str = './data/mathvista/images',
    output_path: str = './results/mathvista_output.json',
    num_gpus: int = 8,
    batch_size: int = 8,
    top_k: int = 40,
    top_p: float = 0.8,
    temperature: float = 0.3,
    max_new_tokens: int = 2048,
    repetition_penalty: float = 1.0,
    seed: int = 3407
):
    pipe = pipeline(
        model_path,
        chat_template_config=ChatTemplateConfig(model_name=model_type),
        backend_config=TurbomindEngineConfig(tp=num_gpus)
    )
    gen_config = GenerationConfig(
        random_seed=seed,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
    )

    os.makedirs(output_path[:output_path.rindex('/')], exist_ok=True)
    if dataset_name == 'mathvista':
        with open(data_path, 'r', encoding='utf-8') as f:
            ori_data = json.load(f)

        all_queries, all_responses, all_extractions = [], [], []
        batch_inputs = []
        for i in tqdm(range(1000)):
            question = ori_data[f'{i + 1}']['question']
            image_path = os.path.join(
                image_root, ori_data[f'{i + 1}']['image'])
            prompt = USER_HINT.format(question=question)
            all_queries.append(prompt)

            if ori_data[f'{i + 1}']['image']:
                prompt = f'{IMAGE_TOKEN}\n{prompt}'
                image = [load_image(image_path).convert('RGB')]
                cur_prompt = (prompt, image)
            else:
                cur_prompt = (prompt)
            batch_inputs.append(cur_prompt)

            if (i + 1) % batch_size == 0 or i + 1 == len(ori_data):
                raw_responses = pipe(batch_inputs, gen_config=gen_config)
                responses = [r.text for r in raw_responses]
                extractions = [extract_answer_try_all_methods(
                    r.text) for r in raw_responses]
                all_responses += responses
                all_extractions += extractions
                batch_inputs = []

                for j in range(i + 1):
                    ori_data[f'{j + 1}']['query'] = all_queries[j]
                    ori_data[f'{j + 1}']['response'] = all_responses[j]
                    ori_data[f'{j + 1}']['extraction'] = all_extractions[j]

                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(ori_data, f, indent=4, ensure_ascii=False)

    elif dataset_name == 'mathverse':
        with open(data_path, 'r', encoding='utf-8') as f:
            ori_data = json.load(f)

        all_responses, all_extractions = [], []
        batch_inputs = []
        for i, d in tqdm(enumerate(ori_data), total=len(ori_data)):
            question = d['question']
            image_path = os.path.join(image_root, d['image'])
            # prompt = USER_HINT.format(question=question)
            prompt = d['query_cot']

            if d['image']:
                prompt = f'{IMAGE_TOKEN}\n{prompt}'
                image = [load_image(image_path).convert('RGB')]
                cur_prompt = (prompt, image)
            else:
                cur_prompt = (prompt)
            batch_inputs.append(cur_prompt)

            if (i + 1) % batch_size == 0 or i + 1 == len(ori_data):
                raw_responses = pipe(batch_inputs, gen_config=gen_config)
                responses = [r.text for r in raw_responses]
                extractions = [extract_answer_try_all_methods(
                    r.text) for r in raw_responses]
                all_responses += responses
                all_extractions += extractions
                batch_inputs = []

                for j, d in enumerate(ori_data):
                    d['model_answer'] = all_responses[j]
                    d['extraction'] = all_extractions[j]
                    if j >= i:
                        break

                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(ori_data, f, indent=4, ensure_ascii=False)

    elif dataset_name == 'math-vision':
        ori_data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                d = json.loads(line)
                ori_data.append(d)

        all_responses, all_extractions = [], []
        batch_inputs = []
        for i, d in tqdm(enumerate(ori_data), total=len(ori_data)):
            question = d['question']
            image_path = os.path.join(image_root, d['image'])
            prompt = USER_HINT.format(question=question)

            prompt = f'{IMAGE_TOKEN}\n{prompt}'
            image = [load_image(image_path).convert('RGB')]

            cur_prompt = (prompt, image)
            batch_inputs.append(cur_prompt)

            if (i + 1) % batch_size == 0 or i + 1 == len(ori_data):
                raw_responses = pipe(batch_inputs, gen_config=gen_config)
                responses = [r.text for r in raw_responses]
                extractions = [extract_answer_try_all_methods(
                    r.text) for r in raw_responses]
                all_responses += responses
                all_extractions += extractions
                batch_inputs = []

                for j, d in enumerate(ori_data):
                    d['model'] = model_type
                    d['response'] = all_responses[j]
                    d['model_answer'] = all_extractions[j]
                    if j >= i:
                        break

                with open(output_path, 'w', encoding='utf-8') as f:
                    for d in ori_data:
                        json.dump(d, f, ensure_ascii=False)
                        f.write('\n')

    elif dataset_name == 'we-math':
        with open(data_path, 'r', encoding='utf-8') as f:
            ori_data = json.load(f)

        base_prompt = """Now, we require you to solve a multiple-choice math question. We will provide
        you with the relevant knowledge concepts of this question for your reference.
        Please briefly describe your thought process and provide the final answer(option).
        Knowledge concept: {knowledge_concept}
        Question: {question}
        Option: {option}
        Regarding the format, please answer following the template below, and be
        sure to include two <> symbols:
        <Thought process>: <<your thought process>> <Answer>: <<your option>>""".replace('\n', ' ')

        all_responses, all_extractions = [], []
        batch_inputs = []
        for i, d in tqdm(enumerate(ori_data), total=len(ori_data)):
            question = d['question']
            option = d['option']
            knowledge_concept = d['knowledge concept description']
            image_path = os.path.join(image_root, d['image_path'])
            prompt = base_prompt.format(
                question=question, option=option, knowledge_concept=knowledge_concept)

            if d['image_path']:
                prompt = f'{IMAGE_TOKEN}\n{prompt}'
                image = [load_image(image_path).convert('RGB')]
                cur_prompt = (prompt, image)
            else:
                cur_prompt = (prompt)
            batch_inputs.append(cur_prompt)

            if (i + 1) % batch_size == 0 or i + 1 == len(ori_data):
                raw_responses = pipe(batch_inputs, gen_config=gen_config)
                responses = [r.text for r in raw_responses]
                extractions = [extract_answer_try_all_methods(
                    r.text) for r in raw_responses]
                all_responses += responses
                all_extractions += extractions
                batch_inputs = []

                for j, d in enumerate(ori_data):
                    d['response'] = all_responses[j]
                    d['extraction'] = all_extractions[j]
                    if j >= i:
                        break

                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(ori_data, f, indent=4, ensure_ascii=False)

    elif dataset_name == 'geoqa':
        def assemble_batch(batch: List[Dict]) -> List[tuple]:
            ret = []
            for item in batch:
                inst = IMAGE_TOKEN + item['question']
                img = item['image']
                instance = (inst, [img])
                ret.append(instance)
            return ret

        label2id = {
            0: 'A',
            1: 'B',
            2: 'C',
            3: 'D',
        }

        with open(data_path, 'rb') as f:
            test_data = pickle.load(f)
        data = []
        for item in test_data:
            image = Image.fromarray(item['image']).convert('RGB')
            choices = [f'{c}. {choice}' for c,
                       choice in zip('ABCD', item['choices'])]
            question = item['subject'] + '\n' + ' '.join(choices)
            answer = label2id[item['label']]
            new_item = {
                'image': image,
                'question': question,
                'answer': answer,
            }
            data.append(new_item)

        len_data = len(data)
        iter_num = int(math.ceil(len_data / batch_size))

        ret_list = []
        for i in tqdm(range(iter_num)):
            start, end = i * batch_size, (i + 1) * batch_size
            batch = assemble_batch(data[start:end])
            responses = pipe(batch, gen_config=gen_config)
            for response in responses:
                ret_list.append({"model_answer": response.text})

        data = [{**i, **j} for i, j in zip(data, ret_list)]
        for item in data:
            if 'image' in item:
                del item['image']
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    else:
        raise NotImplementedError


if __name__ == '__main__':
    fire.Fire(inference)
