import os
from typing import Dict
import transformers
from transformers import Trainer
from train.vta_trainer import VTATrainer
from train.deepseek_math.dataset import SupervisedDataset, DataCollatorForSupervisedDataset
from transformers import AutoModelForCausalLM
from models.deepseek_math import VLChatProcessor, MultiModalityCausalLM
from train.args import ModelArguments, DataArguments, TrainingArguments
from loguru import logger

local_rank = None


def rank0_logger_info(*args):
    if local_rank == 0:
        logger.info(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, model_dir: str, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
        # copy preprocessor_config.json
        ori_path = os.path.join(model_dir, 'preprocessor_config.json')
        os.system(f'cp "{ori_path}" "{output_dir}"')


def make_supervised_data_module(
        vl_chat_processor: VLChatProcessor,
        data_args: DataArguments,
        use_kl: bool = False,
        **kwargs
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    rank0_logger_info(f'Preparing train_dataset...')
    train_dataset = SupervisedDataset(
        data_args=data_args,
        vl_chat_processor=vl_chat_processor,
        split="train",
        use_kl=use_kl,
    )

    rank0_logger_info(f'Preparing data_collator...')
    data_collator = DataCollatorForSupervisedDataset(vl_chat_processor=vl_chat_processor, use_kl=use_kl)

    return dict(
        train_dataset=train_dataset,
        data_collator=data_collator,
    )


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    rank0_logger_info(model_args)
    rank0_logger_info(data_args)
    rank0_logger_info(training_args)

    model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_args.model_name_or_path)
    tokenizer = vl_chat_processor.tokenizer

    data_module = make_supervised_data_module(vl_chat_processor=vl_chat_processor, data_args=data_args,
                                              use_kl=training_args.use_kl)

    if training_args.trainable_parts == "all":
        for n, p in model.named_parameters():
            p.requires_grad = True
    else:
        trainable_parts_list = [p.strip() for p in training_args.trainable_parts.split(',')]
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
