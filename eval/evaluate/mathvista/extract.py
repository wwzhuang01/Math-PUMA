import math
import argparse
import traceback
import concurrent.futures
from openai import OpenAI
from eval.evaluate.eval_utils import *
from eval.evaluate.prompts import mathvista_prompt_score

client = OpenAI(
    api_key='sk-example_key',
    base_url='example_url'
)


def verify_extraction(extraction):
    extraction = extraction.strip()
    if extraction == "" or extraction is None:
        return False
    return True


def create_test_prompt(demo_prompt, query, response):  # few
    demo_prompt = demo_prompt.strip()
    test_prompt = f"{query}\n\n{response}"
    full_prompt = f"{demo_prompt}\n\n{test_prompt}\n\nExtracted answer: "
    return full_prompt


def extract_answer(response, problem, quick_extract=False, llm_engine='gpt-3.5-turbo'):
    question_type = problem['question_type']
    answer_type = problem['answer_type']
    choices = problem['choices']
    query = problem['query']

    if response == "":
        return ""

    if question_type == 'multi_choice' and response in choices:
        return response

    if answer_type == "integer":
        try:
            extraction = int(response)
            return str(extraction)
        except:
            pass

    if answer_type == "float":
        try:
            extraction = str(float(response))
            return extraction
        except:
            pass

    # quick extraction
    if quick_extract:
        print("Quickly extracting answer...")
        try:
            result = re.search(r'The answer is "(.*)"\.', response)
            if result:
                extraction = result.group(1)
                return extraction
        except:
            pass

    # general extraction
    try:
        full_prompt = create_test_prompt(
            mathvista_prompt_score, query, response)
        extraction = get_chat_response_ours(full_prompt, model=llm_engine)
        extraction = extraction.lower()
        if 'extracted answer:' in extraction:
            extraction = ''.join(extraction.split(
                'extracted answer:')[1:]).strip()
        else:
            extraction = extraction.strip()
        return extraction
    except Exception as e:
        print(e)
        print(f"Error in extracting answer for {pid}")

    return response


def extract_answer_quick(response, problem, quick_extract=False):
    question_type = problem['question_type']
    answer_type = problem['answer_type']
    choices = problem['choices']
    query = problem['query']

    if response == "":
        return ""

    if question_type == 'multi_choice' and response in choices:
        return response

    if answer_type == "integer":
        try:
            extraction = int(response)
            return str(extraction)
        except:
            pass

    if answer_type == "float":
        try:
            extraction = str(float(response))
            return extraction
        except:
            pass

    # quick extraction
    if quick_extract:
        print("Quickly extracting answer...")
        try:
            result = response.split('The answer is ')
            if result:
                # extraction = result.group(1)
                extraction = result[1]
                return extraction
        except:
            pass
    return response


def get_chat_response_ours(prompt, model="gpt-4o-mini", temperature=0, max_tokens=256, n=1, patience=3, sleep_time=0):
    messages = [
        {"role": "user", "content": prompt},
    ]
    while patience > 0:
        patience -= 1
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n
            )
            if n == 1:
                prediction = response.choices[0].message.content.strip()
                if prediction != "" and prediction != None:
                    return prediction
            else:
                prediction = [choice['message']['content'].strip()
                              for choice in response.choices]
                if prediction[0] != "" and prediction[0] != None:
                    return prediction

        except Exception as e:
            if "Rate limit" not in str(e):
                print(f'[get_chat_response_ours] {e}')
                traceback.print_exc()

            if "Please reduce the length of the messages" in str(e):
                print("!!Reduce promot size")
                # reduce input prompt and keep the tail
                new_size = int(len(prompt) * 0.9)
                new_start = len(prompt) - new_size
                prompt = prompt[new_start:]
                messages = [
                    {"role": "user", "content": prompt},
                ]

            if sleep_time > 0:
                time.sleep(sleep_time)
    return ""


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # input
    parser.add_argument('--output_dir', type=str,
                        default='./mathvista_outputs')
    parser.add_argument('--output_file', type=str, default='responses.json')
    parser.add_argument('--response_label', type=str, default='response',
                        help='response label for the input file')
    # model
    parser.add_argument('--llm_engine', type=str,
                        default='gpt-3.5-turbo', help='llm engine')
    parser.add_argument('--number', type=int, default=-1,
                        help='number of problems to run')
    parser.add_argument('--quick_extract', default=False,
                        help='use rules to extract answer for some problems')
    parser.add_argument('--rerun', action='store_true',
                        help='rerun the answer extraction')
    # output
    parser.add_argument('--save_every', type=int,
                        default=10, help='save every n problems')
    parser.add_argument('--output_label', type=str,
                        default='', help='label for the output file')
    parser.add_argument('--num_workers', type=int,
                        default=30, help='num_workers')
    args = parser.parse_args()

    # args
    label = args.response_label
    result_file = os.path.join(args.output_dir, args.output_file)

    if args.output_label != '':
        output_file = result_file.replace(
            '.json', f'_{args.output_label}.json')
    else:
        output_file = result_file

    # read results
    print(f"Reading {result_file}...")
    results = read_json(result_file)

    # full pids
    full_pids = list(results.keys())
    if args.number > 0:
        full_pids = full_pids[:min(args.number, len(full_pids))]
    print("Number of testing problems:", len(full_pids))

    # test pids
    if args.rerun:
        test_pids = full_pids
    else:
        test_pids = []
        for pid in full_pids:
            if 'extraction' not in results[pid] or not verify_extraction(results[pid]['extraction']):
                test_pids.append(pid)

    test_num = len(test_pids)
    print("Number of problems to run:", test_num)

    # tqdm, enumerate results
    def run(thread_id: int, start_idx: int, end_idx: int):
        part_pids = test_pids[start_idx:end_idx]
        for i, pid in enumerate(tqdm(part_pids, desc=f'Thread-{thread_id}')):
            problem = results[pid]

            assert label in problem
            response = problem[label]

            extraction = extract_answer(
                response, problem, args.quick_extract, args.llm_engine)
            results[pid]['extraction'] = extraction

    num_workers = args.num_workers
    per_batch_size = int(math.ceil(len(results) / num_workers))
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(run, i, per_batch_size * i, min(per_batch_size * (i + 1), len(results)))
                   for i in range(num_workers)]
        concurrent.futures.wait(futures)

    # results = sorted(results, key=lambda x: int(x[0]))

    print(f"Saving results to {output_file}...")
    save_json(results, output_file)
    print(f"Results saved.")
