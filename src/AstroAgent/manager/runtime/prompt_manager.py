# core/prompt.py

import os
from typing import Dict, Any, List, Tuple

from jinja2 import (
    Environment,
    FileSystemLoader,
    StrictUndefined,
    TemplateError,
)

from AstroAgent.core.config.prompt_config import PromptConfig


class PromptManager:
    """
    Jinja2-based Prompt Manager

    Responsibilities:
    - Load system.md / user.md
    - Render with Jinja2
    - Enforce variable whitelist
    - Provide clear error context (agent, function)
    """

    def __init__(self, config: PromptConfig):
        self.config = config
        self.prompt_root = self.config.root

        self.env = Environment(
            loader=FileSystemLoader(self.prompt_root),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def load(
        self,
        state: Dict[str, Any],
        agent_name: str,
        function_name: str,
        **kwargs,
    ) -> Tuple[str, str]:
        """
        Load and render (system_prompt, user_prompt)
        """

        # 用于错误上下文
        self.agent_name = agent_name
        self.function_name = function_name

        func_cfg = self.config.get_function(agent_name, function_name)

        base_path = self._resolve_path(func_cfg.path)

        system_path = os.path.join(base_path, "system.md")
        user_path = os.path.join(base_path, "user.md")

        if not os.path.isfile(system_path):
            raise FileNotFoundError(
                f"[PromptManager] system.md not found: {system_path}"
            )

        if not os.path.isfile(user_path):
            raise FileNotFoundError(
                f"[PromptManager] user.md not found: {user_path}"
            )

        values = self._collect_variables(
            allowed=func_cfg.variables,
            state=state,
            kwargs=kwargs,
        )

        try:
            system_text = self._render(system_path, values)
            user_text = self._render(user_path, values)
        except TemplateError as e:
            raise RuntimeError(
                f"Prompt render failed "
                f"(agent={agent_name}, function={function_name})"
            ) from e

        return system_text, user_text

    # ------------------------
    # Internal helpers
    # ------------------------

    def _render(self, full_path: str, values: Dict[str, Any]) -> str:
        """
        Render a template file using Jinja2
        """
        # Jinja2 的 loader 是基于 prompt_root 的
        relative_path = os.path.relpath(full_path, self.prompt_root)
        template = self.env.get_template(relative_path)
        return template.render(**values)

    def _resolve_path(self, relative_path: str) -> str:
        return os.path.join(self.prompt_root, relative_path)

    def _collect_variables(
        self,
        allowed: List[str],
        state: Dict[str, Any],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Collect variables strictly from whitelist
        Priority: kwargs > state
        """
        values: Dict[str, Any] = {}

        # 允许 variables 为空列表
        if not allowed:
            return values

        for name in allowed:
            if name in kwargs:
                values[name] = kwargs[name]
            elif name in state:
                values[name] = state[name]
            else:
                raise KeyError(
                    f"Prompt variable '{name}' not found in kwargs or state. "
                    f"agent={self.agent_name}, function={self.function_name}"
                )

        return values
