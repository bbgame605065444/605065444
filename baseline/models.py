"""
SemEval 2026 Task 12: Abductive Event Reasoning
Baseline模型实现 - 支持多种LLM
"""
import os
from abc import ABC, abstractmethod
from typing import List, Set, Optional, Dict
from dataclasses import dataclass


@dataclass
class AERPrompt:
    """AER任务的Prompt"""
    target_event: str
    options: Dict[str, str]
    context: Optional[str] = None


class BaseModel(ABC):
    """基础模型类"""

    @abstractmethod
    def predict(self, prompt: AERPrompt) -> str:
        """返回模型的原始输出"""
        pass

    def format_prompt(self, prompt: AERPrompt, include_context: bool = True) -> str:
        """格式化Prompt"""
        system_prompt = """You are an expert in causal reasoning. Given an observed event and context documents, identify the most plausible and direct cause(s) from the given options.

Rules:
1. Select the option(s) that represent the most DIRECT and PLAUSIBLE cause of the target event
2. You may select multiple options if multiple causes are equally direct and plausible
3. Consider the temporal order: the cause must happen BEFORE the effect
4. Distinguish between direct causes and indirect/background factors
5. Base your reasoning on the provided context documents when available

Output format: Answer with ONLY the letter(s) of your choice, separated by commas if multiple. Example: "A" or "A, B"
"""
        user_prompt = f"Target Event: {prompt.target_event}\n\n"

        if include_context and prompt.context:
            # 限制上下文长度
            max_context_len = 4000
            context = prompt.context[:max_context_len]
            if len(prompt.context) > max_context_len:
                context += "... [truncated]"
            user_prompt += f"Context Documents:\n{context}\n\n"

        user_prompt += "Options:\n"
        for opt, text in prompt.options.items():
            user_prompt += f"{opt}. {text}\n"

        user_prompt += "\nWhich option(s) represent the most direct cause of the target event? Answer:"

        return system_prompt, user_prompt


class OpenAIModel(BaseModel):
    """OpenAI GPT模型"""

    def __init__(self, model_name: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("请安装openai: pip install openai")

    def predict(self, prompt: AERPrompt) -> str:
        system_prompt, user_prompt = self.format_prompt(prompt)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=50
        )

        return response.choices[0].message.content.strip()


class AnthropicModel(BaseModel):
    """Anthropic Claude模型"""

    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("请安装anthropic: pip install anthropic")

    def predict(self, prompt: AERPrompt) -> str:
        system_prompt, user_prompt = self.format_prompt(prompt)

        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=50,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )

        return response.content[0].text.strip()


class HuggingFaceModel(BaseModel):
    """HuggingFace本地模型（适合在本地GPU上运行）"""

    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct", device: str = "auto"):
        self.model_name = model_name
        self.device = device

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=device
            )
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=50
            )
        except ImportError:
            raise ImportError("请安装transformers: pip install transformers torch")

    def predict(self, prompt: AERPrompt) -> str:
        system_prompt, user_prompt = self.format_prompt(prompt)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        outputs = self.pipe(messages, do_sample=False)
        return outputs[0]["generated_text"][-1]["content"].strip()


class OllamaModel(BaseModel):
    """Ollama本地模型（轻量级本地部署）"""

    def __init__(self, model_name: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url

    def predict(self, prompt: AERPrompt) -> str:
        import requests

        system_prompt, user_prompt = self.format_prompt(prompt)

        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "stream": False,
                "options": {"temperature": 0}
            }
        )

        return response.json()["message"]["content"].strip()


class VLLMModel(BaseModel):
    """vLLM高性能推理（适合大规模评测）"""

    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        self.model_name = model_name

        try:
            from vllm import LLM, SamplingParams
            self.llm = LLM(model=model_name)
            self.sampling_params = SamplingParams(temperature=0, max_tokens=50)
        except ImportError:
            raise ImportError("请安装vllm: pip install vllm")

    def predict(self, prompt: AERPrompt) -> str:
        system_prompt, user_prompt = self.format_prompt(prompt)

        full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"

        outputs = self.llm.generate([full_prompt], self.sampling_params)
        return outputs[0].outputs[0].text.strip()

    def batch_predict(self, prompts: List[AERPrompt]) -> List[str]:
        """批量预测（vLLM优势）"""
        full_prompts = []
        for prompt in prompts:
            system_prompt, user_prompt = self.format_prompt(prompt)
            full_prompts.append(f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:")

        outputs = self.llm.generate(full_prompts, self.sampling_params)
        return [output.outputs[0].text.strip() for output in outputs]


def get_model(model_type: str, **kwargs) -> BaseModel:
    """工厂函数：获取模型实例"""
    models = {
        "openai": OpenAIModel,
        "anthropic": AnthropicModel,
        "huggingface": HuggingFaceModel,
        "ollama": OllamaModel,
        "vllm": VLLMModel,
    }

    if model_type not in models:
        raise ValueError(f"不支持的模型类型: {model_type}. 可选: {list(models.keys())}")

    return models[model_type](**kwargs)
