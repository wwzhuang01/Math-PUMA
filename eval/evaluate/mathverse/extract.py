import copy
import math
import argparse
import regex as re
from eval.evaluate.eval_utils import *
from concurrent.futures import ThreadPoolExecutor, as_completed
from eval.evaluate.prompts import demo_prompt_extract, mathverse_key_step_extraction_prompt


def verify_extraction(extraction):
    extraction = extraction.strip()
    if extraction == "" or extraction is None:
        return False
    return True


def create_test_prompt(extract_prompt, response, inst):
    extract_prompt = extract_prompt.strip()
    test_prompt = f"Model response: '{response}'\nExtracted Answer: "
    full_prompt = f"{extract_prompt}\n\n{test_prompt}"
    return full_prompt


def extract_answer(response, inst, model='gpt-4o', use_cot=False):
    # general extraction
    try:
        if use_cot:
            full_prompt = mathverse_key_step_extraction_prompt.format(
                model_output=response)
            extraction = get_chat_response(full_prompt, model=model)
            match = re.search(r"<extracted>(.*?)</extracted>",
                              extraction, re.DOTALL)
            return match.group(1) if match else ''
        else:
            full_prompt = demo_prompt_extract.format(model_output=response)
            extraction = get_chat_response(full_prompt, model=model)
            return extraction
    except Exception as e:
        print(e)
        print(f"Error in extracting answer for {response}")
        return ''


def trunk_response(response, trunk_length):
    if trunk_length <= 0:
        return response
    else:
        return_res = ' '.join(response.split(' ')[-trunk_length:])
        return return_res


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # input
    parser.add_argument('--model_output_file', type=str, default='output.json')
    parser.add_argument('--save_file', type=str, default='answer.json')
    # output
    parser.add_argument('--llm_engine', type=str,
                        default='gpt-3.5-turbo', help='llm engine')
    parser.add_argument('--save_every', type=int,
                        default=10, help='save every n problems')
    parser.add_argument('--cache', action='store_true', help='cache results')
    parser.add_argument('--trunk_response', type=int,
                        default=-1, help='trunk response to the last n words')
    parser.add_argument('--use_cot_for_mathverse',
                        type=bool, default=False, help='')
    parser.add_argument('--num_workers', type=int,
                        default=30, help='num_workers')
    # args
    args = parser.parse_args()

    # read results
    result_file = args.model_output_file
    print(f"Reading {result_file}...")
    results = read_json(result_file)

    os.makedirs(os.path.dirname(args.save_file), exist_ok=True)
    save_results = []

    # enumerate results
    def run(thread_id: int, start_idx: int, end_idx: int):
        part_results = results[start_idx:end_idx]
        len_part_results = len(part_results)
        ret = []
        for i, inst in enumerate(tqdm(part_results, desc=f'Thread-{thread_id}')):
            save_inst = part_results[i] if i < len_part_results else copy.deepcopy(
                inst)
            if args.cache and 'extraction' in save_inst:
                pass
            else:
                if 'model_answer' in save_inst:
                    response = save_inst['model_answer']
                else:
                    response = ''
                    print(save_inst)
                    # some model may output nothing due to safety
                    print("######### NO MODEL ANSWER ###########")
                response = trunk_response(response, args.trunk_response)

                # extraction = extract_answer(response, save_inst, model=args.llm_engine, use_cot=args.use_cot_for_mathverse)
                extraction = extract_answer(
                    response, save_inst, model=args.llm_engine, use_cot=False)
                save_inst['extraction'] = extraction.replace(
                    'Extracted Answer: ', '').strip()  # sometimes gpt will repeat
                # save_results.append(save_inst)
                ret.append(save_inst)
        return ret

    # num_workers = os.cpu_count()
    num_workers = args.num_workers
    per_batch_size = int(math.ceil(len(results) / num_workers))

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(run, i, per_batch_size * i, min(per_batch_size * (i + 1), len(results)))
                   for i in range(num_workers)]
        for future in as_completed(futures):
            save_results += future.result()

    save_results = sorted(save_results, key=lambda x: int(x['sample_index']))

    print(f"Saving results to {args.save_file}...")
    save_json(save_results, args.save_file)
    print(f"Results saved.")
