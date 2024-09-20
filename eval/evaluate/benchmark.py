import os
import subprocess
from eval.evaluate.eval_utils import format_json


# ["mathverse", "mathvista", "wemath"]
benchmark = os.environ.get("BENCHMARK", "mathvista")

run_name = os.environ.get("RUN_NAME", "mathpuma")
model_type = os.environ.get("MODEL_TYPE", "deepseek-vl")
model_path = os.environ.get("MODEL_PATH", "/mnt/bn/gmma/models/deepseek-math-vl-aligned")
num_gpus = int(os.getenv('ARNOLD_WORKER_GPU', 8))
per_gpu_batch_size = int(os.getenv('PER_GPU_BATCH_SIZE', 2))
llm_engine = os.environ.get("LLM_ENGINE", "gpt-4o-mini")
use_gpt_for_extract = bool(int(os.getenv('USE_GPT_FOR_EXTRACT', 1)))
enable_inference = str(os.environ.get("ENABLE_INFERENCE", "true"))
enable_extract = str(os.environ.get("ENABLE_EXTRACT", "true"))
enable_score = str(os.environ.get("ENABLE_SCORE", "true"))
enable_textonly = str(os.environ.get("ENABLE_TEXTONLY", "true"))
num_workers = int(os.environ.get("NUM_WORKERS", 20))
is_customvlm = str(os.environ.get("IS_CUSTOMVLM", "false"))

work_dir = "./"

if is_customvlm.lower() == 'true':
    start_command = (
        f"python3 ./eval/inference/inference_json_customvlm.py"
        f" --work_dir {work_dir}"
        f" --per_gpu_batch_size {per_gpu_batch_size}"
    )
else:
    start_command = (
        f"python3 ./eval/inference/inference_json.py"
    )

if benchmark.lower() == 'mathverse':
    # testmini
    response_file = os.environ.get("RESPONSE_FILE", f"./results/mathverse/{run_name}_response.json")
    extract_file = response_file.replace('.json', '_extract.json')
    score_file = response_file.replace('.json', '_score.json')

    inference_command = (
        f' --dataset_name "mathverse" '
        f' --data_path "./eval/data/mathverse/testmini.json" '
        f' --model_type "{model_type}" '
        f' --model_path "{model_path}" '
        f' --image_root "./eval/data/mathverse/images" '
        f' --output_path "{response_file}" '
        f' --num_gpus {num_gpus} '
        f' --batch_size {num_gpus * per_gpu_batch_size} '
    )
    if enable_inference.lower() == "true":
        subprocess.run(start_command + inference_command, shell=True)

    if use_gpt_for_extract:
        extract_command = (
            f'python3 ./eval/evaluate/mathverse/extract.py '
            f' --model_output_file "{response_file}" '
            f' --save_file "{extract_file}" '
            f' --llm_engine "{llm_engine}" '
            f' --num_workers {num_workers}'
        )
    else:
        extract_command = (
            f'cp "{response_file}" "{extract_file}" '
        )
    if enable_extract.lower() == 'true':
        subprocess.run(extract_command, shell=True)

    score_command = (
        f'python3 ./eval/evaluate/mathverse/score.py '
        f' --answer_extraction_file "{extract_file}" '
        f' --save_file "{score_file}" '
        f' --llm_engine "{llm_engine}" '
        f' --num_workers {num_workers}'
    )
    if enable_score.lower() == 'true':
        subprocess.run(score_command, shell=True)

    # testmini_text_only
    if enable_textonly.lower() == 'true':
        response_file = response_file.replace('.json', '_text_only.json')
        extract_file = response_file.replace('.json', '_extract.json')
        score_file = response_file.replace('.json', '_score.json')

        inference_command = (
            f' --dataset_name "mathverse" '
            f' --data_path "./eval/data/mathverse/testmini_text_only.json" '
            f' --model_type "{model_type}" '
            f' --model_path "{model_path}" '
            f' --image_root "./eval/data/mathverse/images" '
            f' --output_path "{response_file}" '
            f' --num_gpus {num_gpus} '
            f' --batch_size {num_gpus * per_gpu_batch_size} '
        )
        if enable_inference.lower() == 'true':
            subprocess.run(start_command + inference_command, shell=True)

        if use_gpt_for_extract:
            extract_command = (
                f'python3 ./eval/evaluate/mathverse/extract.py '
                f' --model_output_file "{response_file}" '
                f' --save_file "{extract_file}" '
                f' --llm_engine "{llm_engine}" '
                f' --num_workers {num_workers}'
            )
        else:
            extract_command = (
                f'cp "{response_file}" "{extract_file}" '
            )
        subprocess.run(extract_command, shell=True)

        score_command = (
            f'python3 ./eval/evaluate/mathverse/score.py '
            f' --answer_extraction_file "{extract_file}" '
            f' --save_file "{score_file}" '
            f' --llm_engine "{llm_engine}" '
            f' --num_workers {num_workers}'
        )
        subprocess.run(score_command, shell=True)

elif benchmark.lower() == 'mathvista':
    response_file = os.environ.get("RESPONSE_FILE", f"./results/mathvista/{run_name}_response.json")
    score_file = response_file.replace('.json', '_score.json')

    inference_command = (
        f' --dataset_name "mathvista" '
        f' --data_path "./eval/data/mathvista/testmini.json" '
        f' --model_type "{model_type}" '
        f' --model_path "{model_path}" '
        f' --image_root "./eval/data/mathvista" '
        f' --output_path "{response_file}" '
        f' --num_gpus {num_gpus} '
        f' --batch_size {num_gpus * per_gpu_batch_size} '
    )
    if enable_inference.lower() == "true":
        subprocess.run(start_command + inference_command, shell=True)

    if use_gpt_for_extract:
        extract_command = (
            f'python3 ./eval/evaluate/mathvista/extract.py '
            f' --output_dir "/" '
            f' --output_file "{response_file}" '
            f' --llm_engine "{llm_engine}" '
            f' --num_workers {num_workers}'
            f' --rerun '
        )
        if enable_extract.lower() == 'true':
            subprocess.run(extract_command, shell=True)

    score_command = (
        f'python3 ./eval/evaluate/mathvista/score.py '
        f' --output_dir "/" '
        f' --output_file "{response_file}" '
        f' --score_file "{score_file}" '
        f' --gt_file "./eval/data/mathvista/testmini.json" '
        f' --rerun '
    )
    if enable_score.lower() == 'true':
        subprocess.run(score_command, shell=True)
        format_json(score_file)

elif benchmark.lower() == 'wemath':
    # testmini
    response_file = os.environ.get("RESPONSE_FILE", f"./results/we-math/{run_name}_response.json")

    inference_command = (
        f' --dataset_name "we-math" '
        f' --data_path "./eval/data/we-math/testmini.json" '
        f' --model_type "{model_type}" '
        f' --model_path "{model_path}" '
        f' --image_root "./eval/data/we-math/images" '
        f' --output_path "{response_file}" '
        f' --num_gpus {num_gpus} '
        f' --batch_size {num_gpus * per_gpu_batch_size} '
    )
    if enable_inference.lower() == 'true':
        subprocess.run(start_command + inference_command, shell=True)

    score_file = response_file.replace('.json', '_four_dimensional_metrics.json')
    score_command = (
        f'python3 ./eval/evaluate/wemath/four_dimensional_metrics.py '
        f' --model_name "{run_name}" '
        f' --output_json "{response_file}" '
        f' --main_results_json_path "{score_file}" '
    )
    if enable_score.lower() == 'true':
        subprocess.run(score_command, shell=True)

    score_file = response_file.replace('.json', '_accuracy.json')
    score_command = (
        f'python3 ./eval/evaluate/wemath/accuracy.py '
        f' --model_name "{run_name}" '
        f' --output_json "{response_file}" '
        f' --knowledge_structure_nodes_path "./eval/data/we-math/knowledge_structure_nodes.json"'
        f' --main_results_json_path "{score_file}" '
    )
    if enable_score.lower() == 'true':
        subprocess.run(score_command, shell=True)

else:
    raise NotImplementedError
