from pathlib import Path
from typing import Iterator

from ragna.core import Assistant, PackageRequirement, Source


class Mistral7BInstruct(Assistant):
    @classmethod
    def display_name(cls):
        return "turboderp/Mistral-7B-v0.2-exl2"

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
        config.model_dir = str(Path.home() / "shared/analyst/models" / self.display_name())
        config.prepare()

        self.tokenizer = ExLlamaV2Tokenizer(config)

        model = ExLlamaV2(config)
        cache = ExLlamaV2Cache(model, lazy=True)
        model.load_autosplit(cache)
        self.generator = ExLlamaV2StreamingGenerator(model, cache, self.tokenizer)
        self.generator.set_stop_conditions({self.tokenizer.eos_token})

        self.settings = ExLlamaV2Sampler.Settings()
        self.settings.temperature = 0.0

    def make_prompt(self, prompt: str, sources: list[Source]) -> str:
        return "".join(
            [
                f"<s>[INST] ",
                f"You are a helpful assistant that answers prompts by only using the documents listed below. ",
                f"Each individual document is started pattern <doc> and ended by </doc>. ",
                f"If you can't answer a question based on the sources you are given, just say so. Do not make up information.",
                *[f"<doc> {source.content} </doc>" for source in sources],
                f"Reply with OK if you have understood these instructions.",
                f" [/INST]OK</s>[INST] {prompt} [/INST]",
            ]
        )

    def answer(
        self, prompt: str, sources: list[Source], *, max_new_tokens: int = 256
    ) -> Iterator[str]:
        input_ids = self.tokenizer.encode(
            self.make_prompt(prompt, sources), add_bos=False
        )

        self.generator.begin_stream_ex(input_ids, self.settings)

        for _ in range(max_new_tokens):
            result = self.generator.stream_ex()
            if result["eos"]:
                break
            yield result["chunk"]
