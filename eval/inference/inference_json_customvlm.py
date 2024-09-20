import os
import fire
import json
import math
import pickle
import subprocess
import regex as re
from loguru import logger


def divide_into_batches(num, per):
    step = math.ceil(num / per)
    return [(i, min(i + step, num)) for i in range(0, num, step)]


def inference(
    work_dir: str,
    data_path: str,
    num_gpus: int,
    per_gpu_batch_size: int = 2,
    batch_size: int = 8,    # unused
    *args,
    **kwargs
):
    output_path = kwargs.pop('output_path', '')
    dataset_name = kwargs.pop('dataset_name', '')
    model_path = kwargs.pop('model_path', '')
    model_type = kwargs.pop('model_type', '')
    image_root = kwargs.pop('image_root', '')

    logger.info('preparing batches...')
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

    len_data = len(data)
    batch_indices = divide_into_batches(
        len_data, num_gpus * per_gpu_batch_size)

    processes = []
    base_command = f"python3 ./eval/inference/sub_inference_json_customvlm.py"
    for cur_cuda_device in range(num_gpus):
        for j in range(per_gpu_batch_size):
            total_index = cur_cuda_device * per_gpu_batch_size + j
            start, end = batch_indices[total_index]
            target_path = output_path.replace('.json', f'-{total_index}.jsonl')

            command = (
                f"CUDA_VISIBLE_DEVICES={cur_cuda_device}"
                f" cd {work_dir} &&"
                # f" nohup"
                f" {base_command}"
                f" --start_index {start}"
                f" --end_index {end}"
                f" --cuda_device {cur_cuda_device}"
                f" --dataset_name {dataset_name}"
                f" --data_path {data_path}"
                f" --model_path {model_path}"
                f" --model_type {model_type}"
                f" --image_root {image_root}"
                f" --output_path {target_path}"
                # f" &"
            )
            process = subprocess.Popen(command, shell=True)
            processes.append(process)  # Add this line to track the subprocess
            logger.info(f"Started process for segment {start} to {end}")

    logger.info("All processes started.")

    # Wait for all subprocesses to finish
    for process in processes:  # Add this block to wait for all subprocesses
        process.wait()

    # Merge all sub_jsonl into one json(output_path), then delete all sub_jsonl
    base_filename = os.path.basename(output_path).replace('.json', '')
    pattern = re.compile(f"^{re.escape(base_filename)}-(\\d+)\\.jsonl$")
    base_dir = os.path.dirname(output_path)

    sub_files = []
    for file in os.listdir(base_dir):
        match = pattern.match(file)
        if match:
            num = int(match.group(1))
            sub_files.append((num, os.path.join(base_dir, file)))

    sub_files = sorted(sub_files)
    sub_files.sort()
    sub_files = [file for _, file in sub_files]

    if dataset_name == 'mathvista':
        data = {}
        for file in sub_files:
            with open(file, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    json_data = json.loads(line.strip())
                    pid = json_data['pid']
                    data.update({pid: json_data})
    else:
        data = []
        for file in sub_files:
            print(f'reading {file}')
            with open(file, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    data.append(json.loads(line.strip()))
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    # Delete all sub files
    for file in sub_files:
        os.remove(file)

    logger.info("All sub-jsonl files merged and deleted.")


if __name__ == '__main__':
    fire.Fire(inference)
