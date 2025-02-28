import os
import lazyllm
from lazyllm import (
    pipeline, bind, OnlineEmbeddingModule, Reranker,
    Document, Retriever, fc_register, ReactAgent
)

# 设置文档路径
datapath = "path/to/datasets"

# 读取文档内容，同时指定嵌入模型
documents = Document(
    dataset_path=datapath,
    embed=OnlineEmbeddingModule(
        source="qwen", api_key=lazyllm.config['qwen_api_key']
    ),
    manager=False
)

# 构建 RAG pipeline
with pipeline() as ppl_rag:
    ppl_rag.retriever = Retriever(
        documents,
        "CoarseChunk",
        "bm25_chinese",
        topk=3
    )  # 调用检索组件，分组方式采用 CoarseChunk，相似度计算采用 bm25_chinese，返回最相似度最高的三个节点

    ppl_rag.reranker = (
        Reranker(
            "ModuleReranker",
            model=lazyllm.OnlineEmbeddingModule(type="rerank"),
            topk=1,
            output_format='content',
            join=True
        ) | bind(query=ppl_rag.input)
    )  # 采用 Reranker 组件，对检索结果进行重排序

    ppl_rag.formatter = (
        lambda nodes, query: dict(context_str=nodes, query=query)
    ) | bind(query=ppl_rag.input)


@fc_register("tool")
def search_knowledge_base(query: str):
    """
    This is a traditional Chinese knowledge base, which contains classic chapters.

    Args:
        query (str): The query for searching the knowledge base.
    """
    print("search_knowledge_base called")
    return ppl_rag(query)


model_path = "path/to/distill-internlm2-chat-7b"  # 本地模型路径
prompt = (
    "You are a math problem solver. Provide the final answer in a boxed "
    "format using \\boxed{{answer}}."
)  # 设置提示词

# 调用本地数学问题求解模型
math_expert = (
    lazyllm.TrainableModule(model_path)
    .prompt(lazyllm.ChatPrompter(instruction=prompt))
    .start())

@fc_register("tool")
def math_expert(query: str):
    """
    This is a mathematical tool that can be used to solve mathematical problems.

    Args:
        query (str): The mathematical query.
    """
    print("math_expert called")
    return math_expert(query)


# 注册工具
tools = ["search_knowledge_base", "math_expert"]

# 生成器的提示词
prompt = (
    "You will play the role of an AI Q&A assistant and complete a dialogue task. "
    "In this task, you need to provide your answer based on the given context and question."
)

# 构建最终的 pipeline
with pipeline() as ppl:
    ppl.retriever = ReactAgent(
        lazyllm.OnlineChatModule(
            source="sensenova",
            model="SenseChat-5",
            stream=False,
            api_key=lazyllm.config['qwen_api_key']
        ),
        tools
    )  # 采用 ReactAgent 作为 Retrieval Agent，调用工具 search_knowledge_base 和 math_expert

    ppl.formatter = (
        lambda nodes, query: dict(context_str=nodes, query=query)
    ) | bind(query=ppl.input)  # 将输出内容和 query 合并成为一个字典

    ppl.llm = lazyllm.OnlineChatModule(
        source="sensenova",
        model="DeepSeek-R1",
        stream=False,
        api_key=lazyllm.config['sensenova_api_key'],
        secret_key=lazyllm.config['sensenova_secret_key']
    ).prompt(
        lazyllm.ChatPrompter(prompt, extro_keys=["context_str"])
    )  # 采用 DeepSeek-R1 作为 LLM，结合 query 和工具的输出结果生成最终回答


# 启动 Web 服务器
if __name__ == "__main__":
    lazyllm.WebModule(ppl, port=range(23467, 24000)).start().wait()
