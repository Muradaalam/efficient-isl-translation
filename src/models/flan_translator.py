import torch
import torch.nn as nn

from transformers import AutoModelForSeq2SeqLM
from peft import LoraConfig, TaskType, get_peft_model


class FLANISLTranslator(nn.Module):

    def __init__(
        self,
        model_name="google/flan-t5-xl",
        num_landmarks=543,
        lora_rank=16,
        lora_alpha=32,
        dtype=torch.bfloat16
    ):
        super().__init__()

        self.t5 = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=dtype
        )

        self.hidden_size = self.t5.config.d_model

        self.pose_projection = nn.Linear(
            num_landmarks * 3,
            self.hidden_size
        )

        self.pose_norm = nn.LayerNorm(self.hidden_size)

        config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=["q", "v"],
            bias="none"
        )

        self.t5 = get_peft_model(self.t5, config)

        self.t5.print_trainable_parameters()

    def encode_pose(self, pose, valid_mask):
        pose = pose * valid_mask.unsqueeze(-1)

        B, T, L, C = pose.shape

        pose = pose.reshape(B, T, L * C)

        embeddings = self.pose_projection(pose)
        embeddings = self.pose_norm(embeddings)

        return embeddings.to(self.t5.dtype)

    def forward(self, pose, valid_mask, frame_mask, labels):
        embeddings = self.encode_pose(pose, valid_mask)

        return self.t5(
            inputs_embeds=embeddings,
            attention_mask=frame_mask,
            labels=labels,
            use_cache=False
        )

    @torch.no_grad()
    def generate(
        self,
        pose,
        valid_mask,
        frame_mask,
        max_new_tokens=128,
        num_beams=4
    ):
        embeddings = self.encode_pose(pose, valid_mask)

        return self.t5.generate(
            inputs_embeds=embeddings,
            attention_mask=frame_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams
        )