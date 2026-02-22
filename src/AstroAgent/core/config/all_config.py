import os
from pydantic import BaseModel

from AstroAgent.core.config.batch_config import BatchConfig
from AstroAgent.core.config.io_config import IOConfig
from AstroAgent.core.config.model_config import ModelConfig
from AstroAgent.core.config.mcp_config import MCPConfig
from AstroAgent.core.config.params_config import ParamsConfig
from AstroAgent.core.config.prompt_config import PromptConfig


class AllConfig(BaseModel):
    batch: BatchConfig
    io: IOConfig
    model: ModelConfig
    mcp: MCPConfig
    params: ParamsConfig
    prompt: PromptConfig
    retry_delay: int
    max_tries: int

    @classmethod
    def from_env(cls) -> "AllConfig":
        batch_config = BatchConfig.from_env()
        io_config = IOConfig.from_env()
        model_config = ModelConfig.from_env()
        mcp_config = MCPConfig.from_env()
        params_config = ParamsConfig.from_env()
        prompt_config = PromptConfig.from_json()
        retry_delay = int(os.getenv("RETRY_DELAY") or 180)
        max_tries = int(os.getenv("MAX_TRIES") or 3)

        return cls(
            batch = batch_config,
            io = io_config,
            model = model_config,
            mcp = mcp_config,
            params = params_config,
            prompt = prompt_config,
            retry_delay = retry_delay,
            max_tries = max_tries
        )
