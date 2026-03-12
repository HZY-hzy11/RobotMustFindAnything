# LLM 位置查询模块

## 概述

本模块基于 [SiliconFlow](https://siliconflow.cn/) 提供的 OpenAI-compatible API，实现了一个**基于大语言模型的语义位置推理器**。给定要查找的物体名称和多个候选场景的文字描述，利用 LLM 推断目标物体最可能出现在哪个场景中。

### 特性

- **语义位置推理**：根据场景描述文本，推断物体最可能的位置
- **置信度评估**：返回 0~1 的置信度分数，量化推理的把握程度
- **JSON 结构化输出**：使用 JSON Mode 确保输出格式可靠
- **鲁棒的解析**：支持从不规范的 LLM 响应中提取 JSON 结果
- **可配置模型**：支持切换不同的模型和采样参数

---

## 目录结构

```
rofa/llm/
└── llm_query.py    # LLM 位置查询主模块
```

---

## 快速开始

### 安装依赖

```bash
pip install openai
```

### 配置 API Key

```bash
export SILICONFLOW_API_KEY="your_api_key_here"
```

> API Key 可从 [SiliconFlow 官网](https://siliconflow.cn/) 注册获取。

### 基本用法

```python
from llm_query import LLMQuerier, LLMLocationQuery

# 1. 初始化查询器
querier = LLMQuerier(options={
    "model": "Qwen/Qwen3-8B",      # 模型名称（默认值）
    "temperature": 0.2,              # 采样温度（默认 0.2）
    "max_tokens": 256,               # 最大输出 token 数（默认 256）
})

# 2. 构造查询
query = LLMLocationQuery(
    query="cup",
    far_descriptions=[
        (0, "Kitchen with dining table and cabinets"),
        (1, "Living room with sofa and TV"),
        (2, "Bedroom with a queen-sized bed and soft lighting"),
    ]
)

# 3. 推理
result = querier.predict(query)

# 4. 获取结果
if result:
    print(f"最可能的位置索引: {result.index}")
    print(f"置信度: {result.confidence:.3f}")
```

### 命令行运行

```bash
python llm_query.py --query cup --num_candidates 3
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--query` | `"cup"` | 要查找的物体名称 |
| `--num_candidates` | `3` | 候选场景数量 |

---

## API 参考

### `LLMLocationQuery`

查询对象，封装要查找的物体和候选位置描述。

```python
LLMLocationQuery(query, far_descriptions)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `query` | `str` | 要查找的物体名称 |
| `far_descriptions` | `List[Tuple[int, str]]` | 候选位置列表，每项为 `(index, description)` |

---

### `LLMLocationResult`

单次查询的推理结果。

| 属性 | 类型 | 说明 |
|------|------|------|
| `index` | `int` | 推断的候选索引 |
| `confidence` | `float` | 置信度 (0.0 ~ 1.0) |

---

### `LLMQuerier`

LLM 位置查询器主类。

#### 构造函数

```python
LLMQuerier(options=None)
```

| options 参数 | 类型 | 默认值 | 说明 |
|-------------|------|--------|------|
| `model` | `str` | `"Qwen/Qwen3-8B"` | SiliconFlow 上的模型名称 |
| `temperature` | `float` | `0.2` | 采样温度 |
| `max_tokens` | `int` | `256` | 最大输出 token 数 |
| `base_url` | `str` | `"https://api.siliconflow.cn/v1"` | API 地址 |

#### 方法

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `predict(query)` | `LLMLocationQuery` | `LLMLocationResult \| None` | 执行位置推理 |

---

## 推理流程

```
输入: 物体名称 + 候选场景描述列表
        │
        ▼
┌──────────────────────────────────┐
│  1. 构造系统提示词               │  ← 指定 JSON 输出格式
│     (场景检索助手角色)            │
├──────────────────────────────────┤
│  2. 构造用户提示词               │  ← 拼接物体名称 + 候选描述
├──────────────────────────────────┤
│  3. 调用 SiliconFlow LLM API    │  ← JSON Mode，低温采样
│     (OpenAI-compatible)          │
├──────────────────────────────────┤
│  4. 解析 JSON 响应               │  ← 兜底提取 { } 块
├──────────────────────────────────┤
│  5. 验证 index 合法性            │  ← 确认在候选列表中
│     规范化 confidence             │
└──────────────────────────────────┘
        │
        ▼
  LLMLocationResult(index, confidence)
```

### 提示词策略

- **系统提示词**：设定"场景检索助手"角色，强制 JSON 输出 `{"index": <int>, "confidence": <float>}`
- **用户提示词**：列出物体名称和所有候选场景的 `index` + `description`
- **JSON Mode**：通过 `response_format={"type": "json_object"}` 保证结构化输出
- **低温采样**：默认 `temperature=0.2`，提高输出确定性

---

## 环境变量

| 变量名 | 必填 | 说明 |
|--------|------|------|
| `SILICONFLOW_API_KEY` | 是 | SiliconFlow 平台 API Key |
