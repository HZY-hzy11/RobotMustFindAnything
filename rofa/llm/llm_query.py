#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 位置查询模块
================

基于 SiliconFlow OpenAI-compatible API 的语义位置推理。

用法示例:
    from llm_query import LLMLocationQuery, LLMQuerier
    
    querier = LLMQuerier()
    query = LLMLocationQuery(
        query="cup",
        far_descriptions=[
            (0, "Kitchen with dining table and cabinets"),
            (1, "Living room with sofa and TV"),
        ]
    )
    result = querier.predict(query)
    if result:
        print(f"Found at index {result.index} with confidence {result.confidence}")
"""

from __future__ import annotations

import os
import json
from typing import List, Optional, Tuple
from openai import OpenAI


class LLMLocationResult:
    """
    LLM 位置查询的单个结果。
    
    属性:
        index: prior 索引
        confidence: 模型对结果的置信度 (0.0 ~ 1.0)
    """

    def __init__(self, index: int, confidence: float):
        """
        :param index: prior 在列表中的索引
        :param confidence: 置信度分数 (0.0 ~ 1.0)
        """
        self.index = index
        self.confidence = max(0.0, min(1.0, confidence))

    def __repr__(self) -> str:
        return (
            f"LLMLocationResult(index={self.index}, "
            f"confidence={self.confidence:.3f})"
        )


class LLMLocationQuery:
    """
    LLM 位置查询对象，封装查询物体和候选位置描述。
    """

    def __init__(
        self,
        query: str,
        far_descriptions: List[Tuple[int, str]]
    ):
        """
        :param query: 要查找的物体名称
        :param far_descriptions: 候选位置列表，每项为 (index, description)
                                 其中 description 是该位置的语义描述文本
        """
        self.query = query.strip()
        self.far_descriptions = far_descriptions

    def __repr__(self) -> str:
        return (
            f"LLMLocationQuery(query='{self.query}', "
            f"num_candidates={len(self.far_descriptions)})"
        )


class LLMQuerier:
    """
    基于 LLM 的位置查询器。

    使用 SiliconFlow 提供的 OpenAI-compatible API 进行推理。
    根据候选位置的文字描述，推断目标物体最可能出现在哪个位置。

    配置:
        - api_key: 从环境变量 SILICONFLOW_API_KEY 读取
        - base_url: SiliconFlow API 地址
        - model: 使用的模型，默认为 Qwen/Qwen3-8B

    参数 (options):
        - model (str): 模型名称，默认 "Qwen/Qwen3-8B"
        - temperature (float): 采样温度，默认 0.2
        - max_tokens (int): 最大输出 token 数，默认 256
        - base_url (str): API base URL，默认 "https://api.siliconflow.cn/v1"
    """

    DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
    DEFAULT_MODEL = "Qwen/Qwen3-8B"

    def __init__(self, options: Optional[dict] = None):
        """
        初始化 LLM 查询器。

        :param options: 配置选项字典
        :raises RuntimeError: 未设置 SILICONFLOW_API_KEY 环境变量
        """
        self.options = options if options is not None else {}

        # api_key=""  # 手动设置 硅基流动(siliconflow) API Key
        api_key = os.getenv("SILICONFLOW_API_KEY", "")  # 从环境变量（SILICONFLOW_API_KEY）读取 API Key
        if not api_key:
            raise RuntimeError(
                "Missing SiliconFlow API key. Please set env var SILICONFLOW_API_KEY: export SILICONFLOW_API_KEY=your_api_key_here."
            )

        base_url = self.options.get("base_url", self.DEFAULT_BASE_URL)
        model = self.options.get("model", self.DEFAULT_MODEL)

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = self.options.get("temperature", 0.2)
        self.max_tokens = self.options.get("max_tokens", 256)

        print(
            f"[LLMQuerier] Initialized with model={self.model}, "
            f"base_url={base_url}"
        )

    # =====================================================================
    # 内部方法：提示词构造 & API 调用
    # =====================================================================

    def _build_system_prompt(self) -> str:
        """构造系统提示词。"""
        return (
            "你是一个场景检索助手。给定要找的物体名称，以及多个场景的文字描述，"
            "请判断该物体最可能出现在哪个场景。"
            "只允许从给定候选中选择一个。"
            "用 JSON 输出，且仅输出 JSON，不要输出额外文字。"
            'JSON 格式为：{"index": <int>, "confidence": <float 0~1>}.'
            "confidence 表示你对选择的把握（0~1）。"
        )

    def _build_user_prompt(self, query: str, far_descriptions: List[Tuple[int, str]]) -> str:
        """构造用户提示词。"""
        candidates_text = "\n".join(
            f"- index={idx}\n  description: {desc}"
            for (idx, desc) in far_descriptions
        )
        return (
            f"要找的物体：{query}\n\n"
            f"候选场景如下：\n{candidates_text}\n\n"
            "请返回上述 JSON。"
        )

    def _parse_response_json(self, content: str) -> Optional[dict]:
        """
        从 LLM 响应中解析 JSON 对象。

        支持两种情况：
        1. LLM 直接返回 JSON 对象
        2. LLM 返回包含 JSON 的文本（会做兜底提取）

        :param content: LLM 返回的内容
        :return: 解析后的字典，或 None（解析失败）
        """
        content = (content or "").strip()
        if not content:
            return None

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # 兜底：尝试从文本中提取 JSON
            start = content.find("{")
            end = content.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            try:
                return json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                return None

    def _validate_result(
        self,
        data: dict,
        far_descriptions: List[Tuple[int, str]]
    ) -> Optional[Tuple[int, float]]:
        """
        验证并规范化 LLM 返回的结果。

        :param data: LLM 解析后的 JSON 字典
        :param far_descriptions: 候选列表
        :return: (index, confidence) 或 None（验证失败）
        """
        valid_indices = {i for (i, _) in far_descriptions}

        idx = data.get("index")
        conf = data.get("confidence")

        # 验证 index 有效性
        if not isinstance(idx, int) or idx not in valid_indices:
            return None

        # 规范化 confidence
        try:
            conf_f = float(conf)
        except (TypeError, ValueError):
            conf_f = 0.5

        return (idx, conf_f)

    # =====================================================================
    # 核心推理接口
    # =====================================================================

    def predict(self, query: LLMLocationQuery) -> Optional[LLMLocationResult]:
        """
        根据 far.txt 场景描述，推断目标物体最可能出现的位置。

        调用流程：
        1. 构造系统提示词 & 用户提示词
        2. 调用 SiliconFlow LLM API（JSON Mode）
        3. 解析递推结果，验证合法性
        4. 返回 LLMLocationResult 或 None

        :param query: LLMLocationQuery 对象
        :return: LLMLocationResult 或 None（推理失败）
        :raises RuntimeError: API 调用失败
        """
        if not query.query or not query.far_descriptions:
            return None

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(query.query, query.far_descriptions)

        print(
            f"[LLMQuerier] Querying LLM for '{query.query}' "
            f"({len(query.far_descriptions)} candidates)..."
        )

        # 调用 API
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
                extra_body={"enable_thinking": False},
            )
        except Exception as e:
            print(f"[LLMQuerier] API call failed: {e}")
            raise

        # 解析响应
        content = (resp.choices[0].message.content or "").strip()
        data = self._parse_response_json(content)
        if data is None:
            print(f"[LLMQuerier] Failed to parse JSON response: {content}")
            return None

        # 验证结果
        result = self._validate_result(data, query.far_descriptions)
        if result is None:
            print(f"[LLMQuerier] Result validation failed: {data}")
            return None

        idx, conf = result
        llm_result = LLMLocationResult(idx, conf)
        print(
            f"[LLMQuerier] Result: index={idx}, confidence={conf:.3f}"
        )
        return llm_result


# =========================================================================
# 示例入口
# =========================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="LLM Location Query Demo"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="cup",
        help="Object to find"
    )
    parser.add_argument(
        "--num_candidates",
        type=int,
        default=3,
        help="Number of candidate locations for demo"
    )
    args = parser.parse_args()

    # 初始化查询器
    try:
        querier = LLMQuerier()
    except RuntimeError as e:
        print(f"Error: {e}")
        exit(1)

    # 构造示例候选列表
    candidates = [
        (0, "A modern kitchen with stainless steel appliances, white cabinets, and a granite countertop."),
        (1, "A cozy living room with a beige sofa, wooden coffee table, and a flat-screen TV on the wall."),
        (2, "A bedroom with a queen-sized bed, bedside tables, and soft lighting."),
    ][:args.num_candidates]

    # 构造查询
    query = LLMLocationQuery(query=args.query, far_descriptions=candidates)

    # 执行推理
    result = querier.predict(query)
    if result:
        print(f"\n Found: {result}")
    else:
        print("\n Query failed")
