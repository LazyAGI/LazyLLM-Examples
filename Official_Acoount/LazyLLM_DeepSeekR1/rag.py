import os
import lazyllm
from lazyllm import Document, Retriever, pipeline, bind

home_path = os.path.join(os.getcwd(), 'lazyllm_apply')


# 设置基本的生成器提示词
prompt = (
    'You will play the role of an AI Q&A assistant and complete a dialogue task. '
    'In this task, you need to provide your answer based on the given context and question.'
)

datapath = "path/to/datasets"  # 设置检索文档的路径
documents = Document(
    dataset_path=datapath,
    embed=lazyllm.OnlineEmbeddingModule(source="qwen", api_key=lazyllm.config['qwen_api_key']),
    manager=False
)  # 读取文档内容，同时指定embeded模型

llm_with_rag = lazyllm.OnlineChatModule(
    source="sensenova",
    model="DeepSeek-R1",
    stream=False,
    api_key=lazyllm.config['sensenova_api_key'],
    secret_key=lazyllm.config['sensenova_secret_key']
)  # 构建生成组件，这里采用线上deepseek-r1大模型作为生成器

with pipeline() as ppl:
    ppl.retriever = Retriever(documents, "CoarseChunk", "bm25_chinese", topk=3)  # 调用检索组件
    ppl.formatter = (
        lambda nodes, query: dict(context_str="".join([node.get_content() for node in nodes]), query=query)
    ) | bind(query=ppl.input)  # 将检索结果与原始问题构成字典
    ppl.llm = llm_with_rag.prompt(lazyllm.ChatPrompter(prompt, extro_keys=["context_str"]))  # 调用大模型

result_with_rag = ppl("何为大学")  # 执行pipeline，输入问题

print(f"With rag:{result_with_rag}\n")
