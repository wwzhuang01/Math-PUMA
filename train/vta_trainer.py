import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
from typing import Union
from models.deepseek_math import MultiModalityCausalLM
from models.qwen2 import Qwen2vlmForConditionalGeneration


class VTATrainer(Trainer):
    """
    vision-rich & text-rich alignment trainer
    """

    def extract_valid_logits(self, logits, labels):
        logits_list = [
            logits[i, (label != -100).nonzero().squeeze(1), :]
            for i, label in enumerate(labels)
        ]
        return nn.utils.rnn.pad_sequence(
            logits_list, batch_first=True, padding_value=0.0
        )

    def normalize_logits(self, logits):
        return (logits - logits.mean(dim=-1, keepdim=True)) / (logits.std(dim=-1, keepdim=True) + 1e-6)

    def compute_loss(self, _model, inputs, return_outputs=False):
        model: Union[MultiModalityCausalLM, Qwen2vlmForConditionalGeneration] = self._get_model(_model)

        if isinstance(model._fsdp_wrapped_module, MultiModalityCausalLM):
            with torch.no_grad():
                inputs_ref = inputs[0]
                outputs_ref = model(**inputs_ref)

            inputs_act = inputs[1]
            outputs_act = model(**inputs_act)

        elif isinstance(model._fsdp_wrapped_module, Qwen2vlmForConditionalGeneration):
            inputs_ref, inputs_act = inputs[0], inputs[1]
            pixel_values_ref = inputs_ref.pop('pixel_values', None)
            pixel_values_act = inputs_act.pop('pixel_values', None)

            with torch.no_grad():
                if pixel_values_ref is not None:
                    outputs_ref = model(
                        **inputs_ref, pixel_values=pixel_values_ref)
                else:
                    outputs_ref = model(**inputs_ref)

            if pixel_values_act is not None:
                outputs_act = model(**inputs_act, pixel_values=pixel_values_act)
            else:
                outputs_act = model(**inputs_act)

        else:
            raise NotImplementedError

        logits_ref, logits_act = outputs_ref.logits, outputs_act.logits
        labels_ref, labels_act = inputs_ref['labels'], inputs_act['labels']

        logits_act = self._process_logits(logits_act, labels_act)
        logits_ref = self._process_logits(logits_ref, labels_ref)

        kl_loss = self._compute_kl_loss(logits_act, logits_ref)
        hard_loss = outputs_act.loss

        total_loss = self.args.lambda_kl * kl_loss * (self.args.temperature_kl ** 2) + (1 - self.args.lambda_kl) * hard_loss
        return (total_loss, outputs_act) if return_outputs else total_loss

    def _get_model(self, _model) -> Union[MultiModalityCausalLM, Qwen2vlmForConditionalGeneration]:
        return _model.module if isinstance(_model, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else _model

    def _process_logits(self, logits, labels):
        logits = self.extract_valid_logits(logits, labels)
        return self.normalize_logits(logits)

    def _compute_kl_loss(self, logits_act, logits_ref):
        log_probs_act, log_probs_ref, probs_act, probs_ref = self._get_probs(logits_act, logits_ref)

        inf_mask_act = torch.isinf(logits_act)
        inf_mask_ref = torch.isinf(logits_ref)

        log_probs_act = torch.masked_fill(log_probs_act, inf_mask_act, 0)
        log_probs_ref = torch.masked_fill(log_probs_ref, inf_mask_ref, 0)
        probs_act = torch.masked_fill(probs_act, inf_mask_act, 0)
        probs_ref = torch.masked_fill(probs_ref, inf_mask_ref, 0)

        fkl = F.kl_div(log_probs_ref, probs_act, reduction="batchmean") * (self.args.temperature_kl ** 2)
        rkl = F.kl_div(log_probs_act, probs_ref, reduction="batchmean") * (self.args.temperature_kl ** 2)

        return self.args.alpha_kl * fkl + (1 - self.args.alpha_kl) * rkl

    def _get_probs(self, logits_act, logits_ref):
        log_probs_act = F.log_softmax(logits_act / self.args.temperature_kl, dim=-1)
        log_probs_ref = F.log_softmax(logits_ref / self.args.temperature_kl, dim=-1)
        probs_act = F.softmax(logits_act / self.args.temperature_kl, dim=-1)
        probs_ref = F.softmax(logits_ref / self.args.temperature_kl, dim=-1)

        return log_probs_act, log_probs_ref, probs_act, probs_ref
