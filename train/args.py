from typing import Optional
from dataclasses import dataclass, field
from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="<mathpuma>")


@dataclass
class DataArguments:
    data_path: str = field(default="<mathpuma>")
    lazy_load: bool = field(default=True)
    shuffle_data: bool = field(default=False)
    save_preprocessed_dataset: bool = field(default=False)
    load_preprocessed_dataset: bool = field(default=False)
    preprocessed_dataset_path: str = field(default="<mathpuma>")


@dataclass
class TrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(default=1024)
    overwrite_output_dir: bool = field(default=False)
    output_dir: str = field(default="<mathpuma>")
    trainable_parts: str = field(default='all')
    use_kl: bool = field(default=False)
    alpha_kl: float = field(default=0.5)
    lambda_kl: float = field(default=0.5)
    temperature_kl: float = field(default=1.0)
