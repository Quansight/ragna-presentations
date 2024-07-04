from pathlib import Path
from typing import Iterator

from ragna.core import Assistant, PackageRequirement, Source

class Llama38BInstruct(Assistant):
    @classmethod
    def display_name(cls):
        return "Llama-3-8B-Instruct-exl2"

    @classmethod
    def requirements(cls):
        return [
            PackageRequirement("torch"),
            PackageRequirement("exllamav2"),
        ]

    @classmethod
    def is_available(cls):
        requirements_available = super().is_available()
        if not requirements_available:
            return False

        import torch

        return torch.cuda.is_available()

    def __init__(self):
        super().__init__()
        from exllamav2 import (
            ExLlamaV2,
            ExLlamaV2Cache,
            ExLlamaV2Config,
            ExLlamaV2Tokenizer,
        )
        from exllamav2.generator import ExLlamaV2Sampler, ExLlamaV2StreamingGenerator

        config = ExLlamaV2Config()
        config.model_dir = str(Path.home() / "shared/scipy/rags-to-riches" / self.display_name())
        config.prepare()

        self.tokenizer = ExLlamaV2Tokenizer(config)

        model = ExLlamaV2(config)
        cache = ExLlamaV2Cache(model, lazy=True)
        model.load_autosplit(cache)
        self.generator = ExLlamaV2StreamingGenerator(model, cache, self.tokenizer)
        self.generator.set_stop_conditions({self.tokenizer.eos_token_id, 78191})

        self.settings = ExLlamaV2Sampler.Settings()
        self.settings.temperature = 0.0

    def _make_prompt(self, prompt: str, sources: list[Source]) -> str:
        return "\n".join(
            [
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>",
                f"",
                f"Answer the question based only on the following context:",
                *[source.content for source in sources],
                f"<|eot_id|><|start_header_id|>user<|end_header_id|>",
                f"",
                f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
            ]
        )

    def answer(
        self, prompt: str, sources: list[Source], *, max_new_tokens: int = 256
    ) -> Iterator[str]:
        input_ids = self.tokenizer.encode(
            self._make_prompt(prompt, sources), add_bos=False
        )

        self.generator.begin_stream_ex(input_ids, self.settings)

        for _ in range(max_new_tokens):
            result = self.generator.stream_ex()
            if result["eos"]:
                break
            yield result["chunk"]
