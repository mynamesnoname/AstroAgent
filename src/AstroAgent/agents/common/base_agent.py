import os
import re
import json
import asyncio
import logging

from langchain.agents import create_agent
from AstroAgent.manager.runtime.message_manager import create_message


class BaseAgent:

    agent_name: str = "BaseAgent"

    def __init__(self, runtime):
        self.runtime = runtime

        self._text_agent = None
        self._vis_agent = None

        self._text_model = runtime.get_model("llm")
        self._vis_model = runtime.get_model("vlm")

    # --------------------------
    # Lazy agent creation
    # --------------------------

    async def _ensure_text_agent(self, tools):
        if self._text_agent is None:
            self._text_agent = create_agent(self._text_model, tools)

    async def _ensure_vis_agent(self, tools):
        if self._vis_agent is None:
            self._vis_agent = create_agent(self._vis_model, tools)

    # --------------------------
    # Call
    # --------------------------

    async def call_llm_with_context(
        self,
        system_prompt,
        user_prompt,
        image_path=None,
        parse_json=True,
        description="LLM输出",
        want_tools=True,
    ):

        max_retries = self.runtime.configs.max_tries or 3
        retry_delay = self.runtime.configs.retry_delay or 180

        for attempt in range(max_retries + 1):

            try:
                messages = create_message(
                    system_prompt,
                    user_prompt,
                    image_path
                )

                # ---------------- Mode selection ----------------

                if image_path:
                    if want_tools:
                        tools = await self.runtime.get_tools()
                        await self._ensure_vis_agent(tools)
                        response = await self._vis_agent.ainvoke(
                            {"messages": messages},
                            config={"recursion_limit": 300},
                        )
                        raw_content = response["messages"][-1].content
                    else:
                        tools = []
                        await self._ensure_vis_agent(tools)
                        response = await self._vis_agent.ainvoke(
                            {"messages": messages},
                            config={"recursion_limit": 300},
                        )
                        raw_content = response["messages"][-1].content
                else:
                    if want_tools:
                        tools = await self.runtime.get_tools()
                        await self._ensure_text_agent(tools)
                        response = await self._text_agent.ainvoke(
                            {"messages": messages},
                            config={"recursion_limit": 125},
                        )
                        raw_content = response["messages"][-1].content
                    else:
                        tools = []
                        await self._ensure_text_agent(tools)
                        response = await self._text_agent.ainvoke(
                            {"messages": messages},
                            config={"recursion_limit": 125},
                        )
                        raw_content = response["messages"][-1].content

                # ---------------- JSON parse ----------------

                if parse_json:
                    cleaned = re.sub(
                        r"^```(?:json)?\s*|\s*```$",
                        "",
                        raw_content.strip(),
                        flags=re.IGNORECASE,
                    ).strip()

                    try:
                        return json.loads(cleaned)
                    except Exception:
                        return cleaned

                return raw_content

            except Exception as e:

                error_msg = str(e).lower()

                if (
                    attempt < max_retries
                    and ("rate limit" in error_msg
                         or "insufficient_quota" in error_msg)
                ):
                    logging.warning(
                        f"⏳ {description}遇到限制，{retry_delay}秒后重试..."
                        f" (尝试 {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logging.error(f"❌ {description}失败: {str(e)}")
                    raise
