from asyncio.log import logger
from typing import Sequence, Dict, Union, Tuple
import os
import math
import json
import pickle
import datasets as hf_datasets
from dataclasses import dataclass
from torch.utils.data import Dataset
import PIL.Image
from models.deepseek_math import VLChatProcessor
from models.deepseek_math.processing_vlm import VLChatProcessorOutput
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


class SupervisedDataset(Dataset):
    def __init__(
            self,
            data_args,
            vl_chat_processor: VLChatProcessor,
            split: str = "train",
            use_kl: bool = False,
            **kwargs
    ):
        super(SupervisedDataset, self).__init__()
        self.data_args = data_args
        self.data_path = data_args.data_path.split(',')
        self.vl_chat_processor = vl_chat_processor
        self.use_kl = use_kl
        self.input_ids_list = []
        self.input_ids_label_list = []

        if data_args.lazy_load:
            self.data = self.load_and_process_dataset(split=split)
        else:
            if data_args.load_preprocessed_dataset:
                self.data = self.load_processed_dataset(data_args.preprocessed_dataset_path)
            else:
                self.data = self.load_and_process_dataset(split=split)
                if data_args.save_preprocessed_dataset:
                    self.save_processed_dataset(self.data, data_args.preprocessed_dataset_path)

    def load_processed_dataset(self, preprocessed_dataset_path):
        with open(preprocessed_dataset_path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f'load preprocessed_dataset from {preprocessed_dataset_path}')
        return data

    def save_processed_dataset(self, data, preprocessed_dataset_path):
        os.makedirs(os.path.dirname(preprocessed_dataset_path), exist_ok=True)
        with open(preprocessed_dataset_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f'preprocessed_dataset saved to {preprocessed_dataset_path}')

    def load_single_hfdataset(self, data_path: str, split: str = 'train'):
        if data_path.endswith('.json') or data_path.endswith('.jsonl'):
            hf_dataset = hf_datasets.load_dataset('json', data_files=data_path, split=split)
        elif data_path.endswith('.csv'):
            hf_dataset = hf_datasets.load_dataset('csv', data_files=data_path, split=split)
        else:
            hf_dataset = hf_datasets.load_dataset(data_path)[split]
        return hf_dataset

    def preparing_conversation(self, example, use_kl: bool = False):
        image_url = example.get('image_url', '')
        input = example.get('instruction', '')
        label = example.get('output', '')
        if use_kl:
            image_url_act = example.get('image_url_2', '')
            input_act = example.get('instruction_2', '')
            label_act = example.get('output_2', '')

        if image_url:
            image_file_path = image_url
            image_placeholder_token = '<image_placeholder>'
        else:
            image_file_path = ''
            image_placeholder_token = ''
        conversation = [
            {
                "role": "User",
                "content": f"{image_placeholder_token}{input}",
                "images": [image_file_path]
            },
            {
                "role": "Assistant",
                "content": ""
            }
        ]
        ret_dict = {
            "conversation": conversation, "label": label, "image_file_path": image_file_path
        }

        if use_kl:
            if image_url_act:
                image_file_path_act = image_url_act
                image_placeholder_token_act = '<image_placeholder>'
            else:
                image_file_path_act = ''
                image_placeholder_token_act = ''
            conversation_act = [
                {
                    "role": "User",
                    "content": f"{image_placeholder_token_act}{input_act}",
                    "images": [image_file_path_act]
                },
                {
                    "role": "Assistant",
                    "content": ""
                }
            ]
            ret_dict.update({
                "conversation_act": conversation_act, "label_act": label_act, "image_file_path_act": image_file_path_act
            })
        return ret_dict

    # batch encode with specified batch size
    def batch_encode_with_size(self, tokenizer, texts, batch_size=1024):
        all_input_ids = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encoded_batch = tokenizer.batch_encode_plus(batch_texts, padding=True, truncation=True, return_tensors='pt')
            all_input_ids.extend(encoded_batch['input_ids'].tolist())
        return all_input_ids

    def load_and_process_dataset(self, split: str = 'train'):
        assert self.data_args.lazy_load == True or self.use_kl == False, \
            'temporarily unavailable if lazy_load=False and use_kl=True'

        datasets_list = [self.load_single_hfdataset(path, split) for path in self.data_path]
        hf_dataset = hf_datasets.concatenate_datasets(datasets_list)

        hf_dataset = hf_dataset.map(
            lambda example: self.preparing_conversation(example, self.use_kl),
            batched=False,
            num_proc=1,
            remove_columns=hf_dataset.column_names,
            desc="Preparing conversation and label"
        )

        # shuffle
        if self.data_args.shuffle_data:
            hf_dataset = hf_dataset.shuffle(seed=3407)
        # no shuffle
        else:
            hf_dataset = hf_dataset.map(
                lambda x: {"total_len": len(json.dumps(x["conversation"])) + len(x["label"])},
                batched=False,
                num_proc=1,
                desc="Calculating len(conversation) + len(label)"
            )
            hf_dataset = hf_dataset.sort("total_len", reverse=True)
            hf_dataset = hf_dataset.remove_columns(["total_len"])

        # lazy load
        if self.data_args.lazy_load:
            return hf_dataset

        len_ds = len(hf_dataset)
        sft_format_list = []
        label_list = []
        for item in tqdm(hf_dataset, desc='converting sft format...'):
            sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
                conversations=item['conversation'],
                sft_format=self.vl_chat_processor.sft_format,
                system_prompt=self.vl_chat_processor.system_prompt,
            )
            sft_format_list.append(sft_format)
            label_list.append(item['label'])

        # batch encode
        logger.info('Batch encoding...')
        self.input_ids_list = self.batch_encode_with_size(self.vl_chat_processor.tokenizer, sft_format_list)
        self.input_ids_label_list = self.batch_encode_with_size(self.vl_chat_processor.tokenizer, label_list)

        logger.info('Adding new columns to hf_dataset...')
        assert len(self.input_ids_list) == len(self.input_ids_label_list) == len_ds, f"""
            len(self.input_ids_list): {len(self.input_ids_list)}
            len(self.input_ids_label_list): {len(self.input_ids_label_list)}
            len_ds: {len_ds}
        """
        hf_dataset = hf_dataset.add_column('input_ids', self.input_ids_list)
        hf_dataset = hf_dataset.add_column('input_ids_label', self.input_ids_label_list)

        def process_example(example):
            image_file_path = example["image_file_path"]
            pil_image = PIL.Image.open(image_file_path).convert("RGB")
            pil_images = [pil_image]
            prepare_inputs = self.vl_chat_processor(
                input_ids=example['input_ids'],
                input_ids_label=example['input_ids_label'],
                conversations=example["conversation"],
                label=example["label"],
                images=pil_images,
                force_batchify=False
            ).to('cpu')
            return prepare_inputs

        def process_batch(batch):
            return [process_example(example) for example in tqdm(batch)]

        # Split the dataset into batches
        logger.info('Preparing inputs...')
        num_workers = os.cpu_count() - 8
        batch_size = int(math.ceil((len_ds + num_workers - 1) / num_workers))
        batches = [[hf_dataset[j] for j in range(i, min(i + batch_size, len_ds))] for i in range(0, len_ds, batch_size)]

        # Use ThreadPoolExecutor to process batches
        data = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_batch = {executor.submit(process_batch, batch): batch for batch in batches}
            for future in as_completed(future_to_batch):
                batch_data = future.result()
                data.extend(batch_data)

        # Sort data by the length of input_ids
        logger.info('Sorting data...')

        if not self.data_args.shuffle_data:
            data.sort(key=lambda x: x["input_ids"].shape[0])
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        if self.data_args.lazy_load:
            image_file_path = self.data[i]["image_file_path"]
            if image_file_path:
                pil_image = PIL.Image.open(image_file_path).convert("RGB")
                pil_images = [pil_image]
            else:
                pil_images = []
            prepare_inputs = self.vl_chat_processor(
                conversations=self.data[i]["conversation"],
                label=self.data[i]["label"],
                images=pil_images,
                force_batchify=False
            )

            # merge to list
            if self.use_kl:
                image_file_path_act = self.data[i]["image_file_path_act"]
                if image_file_path_act:
                    pil_image_act = PIL.Image.open(image_file_path_act).convert("RGB")
                    pil_images_act = [pil_image_act]
                else:
                    pil_images_act = []
                prepare_inputs_act = self.vl_chat_processor(
                    conversations=self.data[i]["conversation_act"],
                    label=self.data[i]["label_act"],
                    images=pil_images_act,
                    force_batchify=False
                )
                for k1, k2 in zip(prepare_inputs.keys(), prepare_inputs_act.keys()):
                    prepare_inputs[k1] = [prepare_inputs[k1], prepare_inputs_act[k2]]

            return prepare_inputs
        else:
            return self.data[i]


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    vl_chat_processor: VLChatProcessor
    use_kl: bool

    def __call__(self, instances: Sequence[VLChatProcessorOutput]) -> Union[Dict, Tuple[Dict]]:
        if self.use_kl:
            i_ref, i_act = [], []
            for instance in instances:
                i_ref.append(VLChatProcessorOutput(
                    sft_format=instance['sft_format'][0],
                    input_ids=instance['input_ids'][0],
                    pixel_values=instance['pixel_values'][0],
                    num_image_tokens=instance['num_image_tokens'][0],
                    labels=instance['labels'][0]
                ))
                i_act.append(VLChatProcessorOutput(
                    sft_format=instance['sft_format'][1],
                    input_ids=instance['input_ids'][1],
                    pixel_values=instance['pixel_values'][1],
                    num_image_tokens=instance['num_image_tokens'][1],
                    labels=instance['labels'][1]
                ))
            p_ref = self.vl_chat_processor.batchify(i_ref)
            p_act = self.vl_chat_processor.batchify(i_act)
            return dict(
                input_ids=p_ref.input_ids,
                attention_mask=p_ref.attention_mask,
                pixel_values=p_ref.pixel_values,
                images_seq_mask=p_ref.images_seq_mask,
                images_emb_mask=p_ref.images_emb_mask,
                labels=p_ref.labels,
            ), dict(
                input_ids=p_act.input_ids,
                attention_mask=p_act.attention_mask,
                pixel_values=p_act.pixel_values,
                images_seq_mask=p_act.images_seq_mask,
                images_emb_mask=p_act.images_emb_mask,
                labels=p_act.labels,
            )
        else:
            prepares = self.vl_chat_processor.batchify(instances)
            return dict(
                input_ids=prepares.input_ids,
                attention_mask=prepares.attention_mask,
                pixel_values=prepares.pixel_values,
                images_seq_mask=prepares.images_seq_mask,
                images_emb_mask=prepares.images_emb_mask,
                labels=prepares.labels,
            )
