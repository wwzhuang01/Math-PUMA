from typing import Sequence, Dict, List
import json
import torch
import numpy as np
import datasets as hf_datasets
from dataclasses import dataclass
from torch.utils.data import Dataset
import PIL.Image
from einops import rearrange
from train.args import DataArguments, TrainingArguments
from models.qwen2 import Qwen2vlmProcessor


class LazyQwen2vlmDataset(Dataset):
    def __init__(
        self,
        data_args: DataArguments,
        processor: Qwen2vlmProcessor,
        split: str = "train",
        **kwargs
    ):
        super(LazyQwen2vlmDataset, self).__init__()
        self.data_args = data_args
        self.processor = processor
        self.data = self.load_and_process_dataset(split=split)

    def load_single_hfdataset(self, data_path: str, split: str = 'train'):
        if data_path.endswith('.json') or data_path.endswith('.jsonl'):
            hf_dataset = hf_datasets.load_dataset(
                'json', data_files=data_path, split=split)
        elif data_path.endswith('.parquet'):
            hf_dataset = hf_datasets.load_dataset(
                'parquet', data_files=data_path, split=split)
        elif data_path.endswith('.csv'):
            hf_dataset = hf_datasets.load_dataset(
                'csv', data_files=data_path, split=split)
        else:
            hf_dataset = hf_datasets.load_dataset(data_path)[split]
        return hf_dataset

    def prepare_conversation(self, example):
        image_url = example.get('image_url', '')
        instruction = example.get('instruction', '')
        label = example.get('output', '')

        image_file_path = image_url if image_url else ''
        conversation = [{
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image"},
            ]
        }]
        conversation = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True)
        ret_dict = dict(
            conversation=conversation,
            label=label,
            image_file_path=image_file_path
        )
        return ret_dict

    def load_and_process_dataset(self, split: str = 'train'):
        data_path_list = self.data_args.data_path.split(',')
        datasets_list = [self.load_single_hfdataset(
            path, split) for path in data_path_list]
        hf_dataset = hf_datasets.concatenate_datasets(datasets_list)

        hf_dataset = hf_dataset.map(
            lambda example: self.prepare_conversation(example),
            batched=False,
            num_proc=1,
            remove_columns=hf_dataset.column_names,
            desc="Preparing conversation and label"
        )

        if self.data_args.shuffle_data:
            hf_dataset = hf_dataset.shuffle(seed=3407)
        else:
            hf_dataset = hf_dataset.map(
                lambda x: {"total_len": len(json.dumps(
                    x["conversation"])) + len(x["label"])},
                batched=False,
                num_proc=1,
                desc="Calculating len(conversation) + len(label)"
            )
            hf_dataset = hf_dataset.sort("total_len", reverse=True)
            hf_dataset = hf_dataset.remove_columns(["total_len"])
        return hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        image_file_path = self.data[i]["image_file_path"]
        if image_file_path:
            pil_image = PIL.Image.open(image_file_path).convert("RGB")
            pil_images = [pil_image]
        else:
            pil_images = None

        prepare_inputs = self.processor(
            text=self.data[i]["conversation"],
            images=pil_images,
        )

        tokenized_inputs = prepare_inputs['input_ids']
        pixel_values = prepare_inputs.get('pixel_values', None)
        tokenized_output = self.processor.tokenizer(
            self.data[i]['label'], add_special_tokens=False)['input_ids']

        eos = self.processor.tokenizer.eos_token_id

        input_ids = tokenized_inputs + tokenized_output + [eos]
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(tokenized_inputs) + tokenized_output + [eos]

        return dict(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            labels=labels,
        )


@dataclass
class DataCollatorForQwen2vlmDataset(object):
    """Collate examples for supervised fine-tuning."""
    processor: Qwen2vlmProcessor
    training_args: TrainingArguments

    def pad_sequence(self, seq, max_length, padding_value):
        return seq + [padding_value] * (max_length - len(seq))

    def __call__(self, instances: Sequence[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        input_ids = [instance['input_ids'] for instance in instances]
        pixel_values = [instance['pixel_values'] for instance in instances]
        attention_masks = [instance['attention_mask']
                           for instance in instances]
        labels = [instance['labels'] for instance in instances]

        max_length = min(
            max(len(ids)
                for ids in input_ids), self.training_args.model_max_length
        )

        pad_token_id = self.processor.tokenizer.eos_token_id
        padded_input_ids = [self.pad_sequence(
            ids[:max_length], max_length, pad_token_id) for ids in input_ids]
        padded_attention_masks = [self.pad_sequence(
            mask[:max_length], max_length, 0) for mask in attention_masks]
        padded_labels = [self.pad_sequence(
            lbl[:max_length], max_length, -100) for lbl in labels]

        ret_dict = {
            'input_ids': torch.LongTensor(padded_input_ids),
            'attention_mask': torch.LongTensor(padded_attention_masks),
            'labels': torch.LongTensor(padded_labels),
        }

        try:
            pixel_values = torch.FloatTensor(np.array(pixel_values))
            pixel_values = rearrange(pixel_values, "b n c h w -> (b n) c h w")
            ret_dict.update({'pixel_values': pixel_values})
        except Exception as e:
            pass

        return ret_dict


class VTADataset(Dataset):
    def __init__(
        self,
        data_args: DataArguments,
        processor: Qwen2vlmProcessor,
        split: str,
        training_args: TrainingArguments,
        **kwargs
    ):
        super(VTADataset, self).__init__()
        self.data_args = data_args
        self.training_args = training_args
        self.processor = processor
        self.data = self.load_and_process_dataset(split=split)

    def load_single_hfdataset(self, data_path: str, split: str = 'train'):
        if data_path.endswith('.json') or data_path.endswith('.jsonl'):
            hf_dataset = hf_datasets.load_dataset(
                'json', data_files=data_path, split=split)
        elif data_path.endswith('.parquet'):
            hf_dataset = hf_datasets.load_dataset(
                'parquet', data_files=data_path, split=split)
        elif data_path.endswith('.csv'):
            hf_dataset = hf_datasets.load_dataset(
                'csv', data_files=data_path, split=split)
        else:
            hf_dataset = hf_datasets.load_dataset(data_path)[split]
        return hf_dataset

    def prepare_conversation(self, example):
        image_url = example.get('image_url', '')
        instruction = example.get('instruction', '')
        label = example.get('output', '')
        image_url_act = example.get('image_url_2', '')
        instruction_act = example.get('instruction_2', '')
        label_act = example.get('output_2', '')

        image_file_path = image_url if image_url else ''
        conversation = [{
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image"},
            ]
        }]
        conversation = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True)

        image_file_path_act = image_url_act if image_url_act else ''
        conversation_act = [{
            "role": "user",
            "content": [
                {"type": "text", "text": instruction_act},
                {"type": "image"},
            ]
        }]
        conversation_act = self.processor.apply_chat_template(
            conversation_act, add_generation_prompt=True)

        total_length = len(json.dumps(
            conversation)) + len(json.dumps(conversation_act)) + len(label) + len(label_act)
        if total_length > self.training_args.model_max_length:
            return {
                "conversation": None, "label": None, "image_file_path": None,
                "conversation_act": None, "label_act": None, "image_file_path_act": None
            }

        return {
            "conversation": conversation, "label": label, "image_file_path": image_file_path,
            "conversation_act": conversation_act, "label_act": label_act, "image_file_path_act": image_file_path_act
        }

    def load_and_process_dataset(self, split: str = 'train'):
        data_path_list = self.data_args.data_path.split(',')
        datasets_list = [self.load_single_hfdataset(
            path, split) for path in data_path_list]
        hf_dataset = hf_datasets.concatenate_datasets(datasets_list)

        hf_dataset = hf_dataset.map(
            lambda example: self.prepare_conversation(example),
            batched=False,
            num_proc=1,
            remove_columns=hf_dataset.column_names,
            desc="Preparing conversation and label"
        )
        hf_dataset = hf_dataset.filter(lambda x: x['conversation'] is not None)

        if self.data_args.shuffle_data:
            hf_dataset = hf_dataset.shuffle(seed=3407)
        else:
            hf_dataset = hf_dataset.map(
                lambda x: {"total_len": len(json.dumps(
                    x["conversation"])) + len(x["label"])},
                batched=False,
                num_proc=1,
                desc="Calculating len(conversation) + len(label)"
            )
            hf_dataset = hf_dataset.sort("total_len", reverse=True)
            hf_dataset = hf_dataset.remove_columns(["total_len"])
        return hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        eos = self.processor.tokenizer.eos_token_id
        # ref model
        image_file_path = self.data[i]["image_file_path"]
        if image_file_path:
            pil_image = PIL.Image.open(image_file_path).convert("RGB")
            pil_images = [pil_image]
        else:
            pil_images = None
        prepare_inputs = self.processor(
            text=self.data[i]["conversation"],
            images=pil_images,
        )
        tokenized_inputs = prepare_inputs['input_ids']
        pixel_values = prepare_inputs.get('pixel_values', None)
        tokenized_output = self.processor.tokenizer(
            self.data[i]['label'], add_special_tokens=False)['input_ids']

        input_ids = tokenized_inputs + tokenized_output + [eos]
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(tokenized_inputs) + tokenized_output + [eos]

        # act model
        image_file_path_act = self.data[i]["image_file_path_act"]
        if image_file_path_act:
            pil_image_act = PIL.Image.open(image_file_path_act).convert("RGB")
            pil_images_act = [pil_image_act]
        else:
            pil_images_act = None
        prepare_inputs_act = self.processor(
            text=self.data[i]["conversation_act"],
            images=pil_images_act,
        )
        tokenized_inputs_act = prepare_inputs_act['input_ids']
        pixel_values_act = prepare_inputs_act.get('pixel_values', None)
        tokenized_output_act = self.processor.tokenizer(
            self.data[i]['label_act'], add_special_tokens=False)['input_ids']

        input_ids_act = tokenized_inputs_act + tokenized_output_act + [eos]
        attention_mask_act = [1] * len(input_ids_act)
        labels_act = [-100] * len(tokenized_inputs_act) + \
            tokenized_output_act + [eos]

        return dict(
            input_ids=[input_ids, input_ids_act],
            pixel_values=[pixel_values, pixel_values_act],
            attention_mask=[attention_mask, attention_mask_act],
            labels=[labels, labels_act],
        )


@dataclass
class DataCollatorForVTADataset(object):
    """Collate examples for supervised fine-tuning."""
    processor: Qwen2vlmProcessor
    training_args: TrainingArguments

    def pad_sequence(self, seq, max_length, padding_value):
        return seq + [padding_value] * (max_length - len(seq))

    def __call__(self, instances: Sequence[Dict]) -> Dict:
        pad_token_id = self.processor.tokenizer.eos_token_id

        input_ids = [instance['input_ids'][0] for instance in instances]
        pixel_values = [instance['pixel_values'][0] for instance in instances]
        attention_masks = [instance['attention_mask'][0]
                           for instance in instances]
        labels = [instance['labels'][0] for instance in instances]
        input_ids_act = [instance['input_ids'][1] for instance in instances]
        pixel_values_act = [instance['pixel_values'][1]
                            for instance in instances]
        attention_masks_act = [instance['attention_mask'][1]
                               for instance in instances]
        labels_act = [instance['labels'][1] for instance in instances]

        max_length = min(
            max(len(ids)
                for ids in input_ids), self.training_args.model_max_length
        )
        max_length_act = min(
            max(len(ids)
                for ids in input_ids_act), self.training_args.model_max_length
        )

        # ref
        padded_input_ids = [self.pad_sequence(
            ids[:max_length], max_length, pad_token_id) for ids in input_ids]
        padded_attention_masks = [self.pad_sequence(
            mask[:max_length], max_length, 0) for mask in attention_masks]
        padded_labels = [self.pad_sequence(
            lbl[:max_length], max_length, -100) for lbl in labels]

        try:
            pixel_values = torch.FloatTensor(np.array(pixel_values))
            pixel_values = rearrange(pixel_values, "b n c h w -> (b n) c h w")
        except Exception as e:
            pixel_values = None

        # act
        padded_input_ids_act = [self.pad_sequence(
            ids[:max_length_act], max_length_act, pad_token_id) for ids in input_ids_act]
        padded_attention_masks_act = [self.pad_sequence(
            mask[:max_length_act], max_length_act, 0) for mask in attention_masks_act]
        padded_labels_act = [self.pad_sequence(
            lbl[:max_length_act], max_length_act, -100) for lbl in labels_act]

        try:
            pixel_values_act = torch.FloatTensor(np.array(pixel_values_act))
            pixel_values_act = rearrange(
                pixel_values_act, "b n c h w -> (b n) c h w")
        except Exception as e:
            pixel_values_act = None

        return {
            'input_ids': torch.LongTensor(padded_input_ids),
            'pixel_values': pixel_values,
            'attention_mask': torch.LongTensor(padded_attention_masks),
            'labels': torch.LongTensor(padded_labels),
        }, {
            'input_ids': torch.LongTensor(padded_input_ids_act),
            'pixel_values': pixel_values_act,
            'attention_mask': torch.LongTensor(padded_attention_masks_act),
            'labels': torch.LongTensor(padded_labels_act),
        }
