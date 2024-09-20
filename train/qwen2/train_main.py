import os
from typing import Dict
from transformers import (
    Trainer,
    HfArgumentParser,
)
from loguru import logger
from train.vta_trainer import VTATrainer
from models.qwen2 import Qwen2vlmProcessor, Qwen2vlmForConditionalGeneration
from train.args import ModelArguments, DataArguments, TrainingArguments
from train.qwen2.dataset import (
    VTADataset,
    DataCollatorForVTADataset,
    LazyQwen2vlmDataset,
    DataCollatorForQwen2vlmDataset
)

local_rank = None


def rank0_logger_info(*args):
    if local_rank == 0:
        logger.info(*args)


def safe_save_model_for_hf_trainer(trainer: Trainer, model_dir: str, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
        ori_path = os.path.join(model_dir, 'preprocessor_config.json')
        os.system(f'cp "{ori_path}" "{output_dir}"')
        ori_path = os.path.join(model_dir, 'processor_config.json')
        os.system(f'cp "{ori_path}" "{output_dir}"')


def make_data_module(
    processor: Qwen2vlmProcessor,
    data_args: DataArguments,
    training_args: TrainingArguments,
    **kwargs
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    rank0_logger_info(f'Preparing train_dataset...')

    if training_args.use_kl:
        dataset_class = VTADataset
        data_collator_class = DataCollatorForVTADataset
    else:
        dataset_class = LazyQwen2vlmDataset
        data_collator_class = DataCollatorForQwen2vlmDataset

    if data_args.lazy_load:
        train_dataset = dataset_class(
            data_args,
            processor,
            split="train",
            training_args=training_args,
        )
    else:
        raise NotImplementedError('Currently only supported lazy_load.')

    rank0_logger_info(f'Preparing data_collator...')
    data_collator = data_collator_class(processor, training_args)

    return dict(
        train_dataset=train_dataset,
        data_collator=data_collator,
    )


def train():
    global local_rank

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    rank0_logger_info(f'ModelArguments:\n{model_args}')
    rank0_logger_info(f'DataArguments:\n{data_args}')
    rank0_logger_info(f'TrainingArguments:\n{training_args}')

    # load model
    rank0_logger_info('loading model...')
    model: Qwen2vlmForConditionalGeneration = Qwen2vlmForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
    processor: Qwen2vlmProcessor = Qwen2vlmProcessor.from_pretrained(model_args.model_name_or_path)
    tokenizer = processor.tokenizer
    data_module = make_data_module(processor, data_args, training_args)

    if training_args.trainable_parts == "all":
        for n, p in model.named_parameters():
            p.requires_grad = True
    else:
        trainable_parts_list = [p.strip() for p in training_args.trainable_parts.split(',')]
        assert len(trainable_parts_list) > 0, "No trainable parts."

        for n, p in model.named_parameters():
            for part in trainable_parts_list:
                if part in n:
                    p.requires_grad = True
                    break
                p.requires_grad = False

    num_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rank0_logger_info(f"trainable parts: {training_args.trainable_parts}")
    rank0_logger_info(f"Number of requires_grad parameters: {num_grad_params}")

    rank0_logger_info('Start training...')
    if training_args.use_kl:
        trainer = VTATrainer(
            model=model, tokenizer=tokenizer, args=training_args, **data_module
        )
    else:
        trainer = Trainer(
            model=model, tokenizer=tokenizer, args=training_args, **data_module
        )
    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, model_dir=model_args.model_name_or_path,
                                   output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
