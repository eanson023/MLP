import os
import torch.nn as nn
import torch.nn.functional as F
from peft import get_peft_model, LoraConfig, PeftModel
from typing import List, Union

from mlp.model.blocks import Conv1D


class RobertaEncoder(nn.Module):
    def __init__(self, modelpath: str,
                 finetune: bool,
                 latent_dim: int = 256,
                 drop_rate: float = 0.0,
                 ckpt_dir: str = None,
                 **kwargs) -> None:
        super(RobertaEncoder, self).__init__()

        self.finetune = finetune
        from transformers import AutoTokenizer, AutoModel
        from transformers import logging
        logging.set_verbosity_error()
        # Tokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = AutoTokenizer.from_pretrained(modelpath)

        # Text model
        self.text_model = AutoModel.from_pretrained(modelpath, add_pooling_layer=False)
        if not finetune:
            self.text_model.training = False
            for p in self.text_model.parameters():
                p.requires_grad = False
        else:
            peft_config = LoraConfig(
                    task_type="none", inference_mode=False, r=8, lora_alpha=16, lora_dropout=drop_rate)
            self.text_model = get_peft_model(self.text_model, peft_config)
            # There is a bug in only storing the weight of LoRA, 
            # so the way of storing the overall Roberta+LoRA weight is still used
            # if ckpt_dir is None:
            #     peft_config = LoraConfig(
            #         task_type="none", inference_mode=False, r=8, lora_alpha=16, lora_dropout=drop_rate)
            #     self.text_model = get_peft_model(self.text_model, peft_config)
            # else:
            #     # checkpoint resume
            #     assert len(ckpt_dir) > 0
            #     print('Load ckpt LoRA....')
            #     # Must set is trainable=True to make LoRA trainable
            #     self.text_model = PeftModel.from_pretrained(self.text_model, ckpt_dir, is_trainable=True)

            print('-------------------- Roberta (Frozen) + LoRA (Trainable) --------------')
            self.text_model.print_trainable_parameters()
            print('-----------------------------------------------------------------------')

        # Then configure the model
        text_encoded_dim = self.text_model.config.hidden_size
        # output linear layer
        self.linear = Conv1D(in_dim=text_encoded_dim, out_dim=latent_dim,
                             kernel_size=1, stride=1, padding=0, bias=True)

    def save_lora(self, ckpt):
        if not self.finetune: return
        # By the way, save the extra lora weight
        self.text_model.save_pretrained(ckpt)
    
    def train(self, mode: bool = True):
        if self.finetune: return
        self.training = mode
        for module in self.children():
            # Don't put the model in
            if module == self.text_model and not self.finetune:
                continue
            module.train(mode)
        return self

    """There is a bug in only storing the weight of LoRA, 
        so the way of storing the overall Roberta+LoRA weight is still used"""
    # def state_dict(self, destination=None, prefix='', keep_vars=False):
    #     state_dict = super().state_dict(destination, prefix, keep_vars)
    #     # Exclude 'text_model' from the state_dict
    #     for key in list(state_dict.keys()):
    #         if key.startswith('text_model'):
    #             state_dict.pop(key)
    #     return state_dict

    # def load_state_dict(self, state_dict):
    #     # Ignore missing text_model weights (already imported on initialization)
    #     # Set strict=False to ignore missing keys
    #     return super().load_state_dict(state_dict, strict=False)

    def get_last_hidden_state(self, texts: List[str], return_mask: bool = False):
        encoded_inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True)
        output = self.text_model(**encoded_inputs.to(self.text_model.device))
        if not return_mask:
            return output.last_hidden_state
        return output.last_hidden_state, encoded_inputs.attention_mask.to(dtype=bool)

    def forward(self, texts: List[str]):
        text_encoded, mask = self.get_last_hidden_state(
            texts, return_mask=True)
        emb_token = self.linear(text_encoded)
        # sentence-level(CLS token), word-level, query_mask
        return emb_token, mask
