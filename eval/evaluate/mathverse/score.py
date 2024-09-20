import copy
import math
import argparse
import regex as re
import pandas as pd
from collections import defaultdict
from eval.evaluate.eval_utils import *
from concurrent.futures import ThreadPoolExecutor, as_completed
from eval.evaluate.prompts import mathverse_prompt_score, mathverse_multi_step_scoring_prompt


def verify_extraction(extraction):
    extraction = extraction.strip()
    if extraction == "" or extraction is None:
        return False
    return True


def create_test_prompt(demo_prompt, inst):
    demo_prompt = demo_prompt.strip()
    full_prompt = demo_prompt.format(
        question=inst['question_for_eval'], gt=inst['answer'], extraction=inst['extraction'])
    return full_prompt


def match_answer(inst, quick_match=False, model='gpt-4o', use_cot: bool = False):
    # quick match
    if quick_match:
        return '1' if inst['answer'] == inst['extraction'] else '0'
    # general extraction
    try:
        if use_cot:
            full_prompt = mathverse_multi_step_scoring_prompt.format(
                question=inst['question_for_eval'], gt=inst['answer'], extraction=inst['extraction']
            )
            extraction = get_chat_response(full_prompt, model=model)
            average_score_match = re.search(
                r"<average_score>(.*?)</average_score>", extraction, re.DOTALL)
            final_answer_score_match = re.search(
                r"<final_answer_score>(.*?)</final_answer_score>", extraction, re.DOTALL)
            if average_score_match and final_answer_score_match:
                try:
                    average_score = float(average_score_match.group(1).strip())
                except:
                    average_score = 0.0
                try:
                    final_answer_score = float(
                        final_answer_score_match.group(1).strip())
                except:
                    final_answer_score = 0.0
                return str(average_score * 0.7 + final_answer_score * 0.3)
            else:
                return ''
        else:
            full_prompt = mathverse_prompt_score.format(
                question=inst['question_for_eval'], gt=inst['answer'], extraction=inst['extraction']
            )
            extraction = get_chat_response(full_prompt, model=model)
            # return extraction.replace("Judgement:", "").strip()
            return extraction.strip()
    except Exception as e:
        print(e)
        print(f"Error in matching answer")

    return ""


def trunk_response(response, trunk_length):
    if trunk_length <= 0:
        return response
    else:
        return_res = ' '.join(response.split(' ')[-trunk_length:])
        return return_res


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # input
    parser.add_argument('--answer_extraction_file',
                        type=str, default='answer.json')
    parser.add_argument('--save_file', type=str, default='answer.json')
    # match
    parser.add_argument('--quick_match', action='store_true',
                        help='use rules to match answer for some problems')
    # output
    parser.add_argument('--llm_engine', type=str,
                        default='gpt-3.5-turbo', help='llm engine')
    parser.add_argument('--save_every', type=int,
                        default=10, help='save every n problems')
    parser.add_argument('--cache', action='store_true', help='cache results')
    parser.add_argument('--trunk_response', type=int,
                        default=-1, help='trunk response to the last n words')
    parser.add_argument('--patience', type=int, default=3,
                        help='patience for runing GPT')
    parser.add_argument('--use_cot_for_mathverse',
                        action='store_true', help='')
    parser.add_argument('--num_workers', type=int,
                        default=30, help='num_workers')

    # args
    args = parser.parse_args()

    if args.use_cot_for_mathverse:
        print(f'using cot for mathverse: {args.use_cot_for_mathverse}')

    # read results
    result_file = args.answer_extraction_file
    print(f"Reading {result_file}...")
    results = read_json(result_file)

    os.makedirs(os.path.dirname(args.save_file), exist_ok=True)
    save_results = []

    score_dict = defaultdict(lambda: defaultdict(list))
    score_version_dict = defaultdict(list)
    # tqdm, enumerate results

    def run(thread_id: int, start_idx: int, end_idx: int):
        part_results = results[start_idx:end_idx]
        len_part_results = len(part_results)
        ret = []

        for i, inst in enumerate(tqdm(part_results, desc=f'Thread-{thread_id}')):
            save_inst = part_results[i] if i < len_part_results else copy.deepcopy(
                inst)
            if args.cache and 'judgement' in save_inst:
                pass
            else:
                patience = args.patience
                while patience > 0:
                    judgement = match_answer(
                        save_inst, quick_match=args.quick_match, model=args.llm_engine, use_cot=False)
                    save_inst['ori_judgement'] = judgement
                    judgement = judgement.lower()
                    if 'judgement:' in judgement:
                        search_target = ''.join(
                            judgement.split('judgement:')[1:]).strip()
                    else:
                        search_target = judgement
                    try:
                        match = re.search(r'(\b\d+(\.\d+)?\b)', search_target)
                        if match:
                            save_inst['judgement'] = float(match.group(0))
                        else:
                            save_inst['judgement'] = 0
                        break
                    except Exception as e:
                        patience -= 1

                if patience <= 0:
                    print(
                        f'\n[{thread_id} - {i}]: reached max patience.\n{judgement}\n', flush=True)
                    save_inst['judgement'] = 0

                ret.append(save_inst)

        return ret

    num_workers = args.num_workers
    per_batch_size = int(math.ceil(len(results) / num_workers))
    futures_results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                run, i, per_batch_size *
                i, min(per_batch_size * (i + 1), len(results))
            ) for i in range(num_workers)
        ]
        for future in as_completed(futures):
            futures_results.append(future.result())

    for res in futures_results:
        for save_inst in res:
            save_results.append(save_inst)
            score_dict[save_inst['metadata']['subject']][save_inst['metadata']
                                                         ['subfield']].append(save_inst['judgement'])
            score_version_dict[save_inst['problem_version']].append(
                save_inst['judgement'])

    save_results = sorted(save_results, key=lambda x: int(x['sample_index']))
    save_json(save_results, args.save_file)

    with open(args.save_file, 'r', encoding='utf-8') as f:
        js_data = json.load(f)
        df = pd.json_normalize(js_data)
        by_problem_version = df.groupby('problem_version')[
            'judgement'].value_counts()
        by_subject = df.groupby('metadata.subject')['judgement'].value_counts()

    with open(args.save_file.replace('.json', '.log'), 'w', encoding='utf-8') as f:
        f.write("=" * 10)
        f.write("By Problem Version:\n")
        f.write(by_problem_version.to_string())
        f.write("\n\nBy Subject:\n")
        f.write(by_subject.to_string())
        f.write("\n\n")

        # subject level acc
        total_cnt, right_cnt = 0, 0
        for subject in score_dict:
            subject_total_cnt, subject_right_cnt = 0, 0
            for subfield in score_dict[subject]:
                subfield_total_cnt = len(score_dict[subject][subfield])
                subfield_right_cnt = len(
                    [inst for inst in score_dict[subject][subfield] if inst == 1])
                subject_total_cnt += subfield_total_cnt
                subject_right_cnt += subfield_right_cnt
                print(
                    f"{subject}-{subfield} Acc: {(subfield_right_cnt/subfield_total_cnt):.3f}")
                f.write(
                    f"{subject}-{subfield} Acc: {(subfield_right_cnt/subfield_total_cnt):.3f}\n")
            print(f"{subject} Acc: {(subject_right_cnt/subject_total_cnt):.3f}")
            f.write(
                f"{subject} Acc: {(subject_right_cnt/subject_total_cnt):.3f}\n")
            total_cnt += subject_total_cnt
            right_cnt += subject_right_cnt
        print(f"Total Acc: {(right_cnt/total_cnt):.3f}")
        f.write(f"Total Acc: {(right_cnt/total_cnt):.3f}\n")

        # version level acc
        total_cnt, right_cnt = 0, 0
        for version in score_version_dict:
            version_total_cnt = len(score_version_dict[version])
            version_right_cnt = len(
                [inst for inst in score_version_dict[version] if inst == 1])
            total_cnt += version_total_cnt
            right_cnt += version_right_cnt
            print(f"{version} Acc: {(version_right_cnt/version_total_cnt):.3f}")
            f.write(
                f"{version} Acc: {(version_right_cnt/version_total_cnt):.3f}\n")
            print(version_total_cnt)
            f.write(f"{version_total_cnt}\n")

        print(f"Acc: {(right_cnt/total_cnt):.3f}")
        f.write(f"Acc: {(right_cnt/total_cnt):.3f}\n\n")
