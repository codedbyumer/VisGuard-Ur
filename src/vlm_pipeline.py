"""Real Vision-Language Model inference for VisGuard-Ur.

Replaces the transparent simulator described in evaluate.py's docstring
with an actual open-source VLM (Qwen2-VL-7B-Instruct by default). This
answers the project's Step 2 research question for real, instead of by
label: does the model's own safety behavior change when the adversarial
instruction is Urdu-script / Roman-Urdu text embedded inside an image?

NOTE: this file could not be executed or network-tested in the
environment that generated it (no internet access there). It follows
the standard Qwen2-VL / Hugging Face `transformers` API. Run
`vlm_evaluate.py --smoke-test` on a handful of images in Colab first
(see the updated README section) before launching the full dataset.
"""
from __future__ import annotations

import torch


class VLMPipeline:
    """Thin wrapper around Qwen2-VL-7B-Instruct for single image+text queries."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-VL-7B-Instruct",
        load_in_4bit: bool = True,
        max_new_tokens: int = 256,
    ) -> None:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.max_new_tokens = max_new_tokens
        quant_kwargs = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )

        print(f"Loading {model_id} (4-bit={load_in_4bit}) ...")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto", **quant_kwargs
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        print("Model loaded.")

    def query(self, image_path: str, instruction: str) -> str:
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0].strip()


# If Qwen2-VL-7B OOMs on a free Colab T4 even at 4-bit, switch
# model_id to "Qwen/Qwen2-VL-2B-Instruct" -- weaker but fits comfortably
# and is still a fair proof-of-concept comparison point.
