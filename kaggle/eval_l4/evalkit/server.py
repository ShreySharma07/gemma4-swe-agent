"""ScorerServer: vLLM 0.19.1 with the scorer's settings, and the agent sandbox made offline."""
import os
import time

from .config import TARGET_MODEL
from .util import log

SCORER_ENV = {"LITELLM_LOCAL_MODEL_COST_MAP": "True", "TRANSFORMERS_NO_TF": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
              "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "1", "VLLM_ENGINE_READY_TIMEOUT_S": "1200",
              "VLLM_NO_USAGE_STATS": "1", "OTEL_SDK_DISABLED": "true",
              "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}


def set_scorer_env():
    os.environ.update(SCORER_ENV)


def make_sandbox_offline():
    """Dead proxy for everything except localhost, like the scorer's network_mode=none."""
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
        os.environ[k] = "http://127.0.0.1:9"
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "localhost,127.0.0.1"
    os.environ["PIP_NO_INDEX"] = "1"


class ScorerServer:
    def __init__(self, model_path, adapters):
        import torch
        from adk_submission import VllmConfig, VllmServer
        self.server = VllmServer(VllmConfig(
            model=model_path, port=8000, host="127.0.0.1", tool_call_parser="gemma4", reasoning_parser="gemma4",
            default_chat_template_kwargs={"enable_thinking": True}, max_model_len=32768,
            dtype="bfloat16" if torch.cuda.is_bf16_supported() else "auto", gpu_memory_utilization=0.80,
            enable_auto_tool_choice=True, enable_lora=True, max_loras=8, max_lora_rank=128,
            tensor_parallel_size=4, startup_timeout=60 * 25), adapter_manifest=adapters)

    def start(self):
        t = time.time()
        self.server.start()
        log(f"vLLM up on {self.server.base_url} after {time.time() - t:.0f}s")
        return self

    def registry(self, declared_model):
        """LiteLLM clients for the declared model name (and the target name) plus one per adapter."""
        return self.server.create_model_registry(aliases=[declared_model, TARGET_MODEL], model_prefix="openai/",
                                                 api_key="EMPTY")

    def stop(self):
        try:
            self.server.stop()
        except Exception:  # noqa: BLE001
            pass
