# -*- coding: utf-8 -*-
# flake8: noqa: F821

import lazyllm
from lazyllm import LOG
from lazyllm import pipeline, parallel, bind, OnlineEmbeddingModule, SentenceSplitter, Document, Retriever, Reranker
from utils.magic_pdf_reader import MagicPDFReader
from utils.magic_pdf_transform import MagicPDFTransform
from utils.utils import draw_pdf_bbox

# 注意：使用前须配置LLM服务的API Key环境变量 如 export LAZYLLM_DEEPSEEK_API_KEY=""、export LAZYLLM_QWEN_API_KEY=""
prompt = 'You will play the role of an AI question-answering assistant and complete a conversation task in which you need to provide your answer based on the given context and question. Please note that if the given context cannot answer the question, do not use your prior knowledge but tell the user that the given context cannot answer the question.'

documents = Document(dataset_path="", embed=OnlineEmbeddingModule(), manager=False)

documents.add_reader("**/*.pdf", MagicPDFReader)  # ​​⚠️​​添加magic-pdf作为pdf文档解析器
documents.create_node_group(name="magic-pdf", transform=MagicPDFTransform)  # ​⚠️​​添加定制节点解析器


with pipeline() as ppl:
    with parallel().sum as ppl.prl:
        prl.retriever1 = Retriever(documents, group_name="magic-pdf", similarity="cosine", topk=3)
        prl.retriever2 = Retriever(documents, group_name="magic-pdf",  similarity="bm25_chinese", topk=3)
    ppl.reranker = Reranker("ModuleReranker", model=OnlineEmbeddingModule(type="rerank"), topk=1) | bind(query=ppl.input)
    ppl.draw_pdf = draw_pdf_bbox | bind(query=ppl.input)    # ​​⚠️ 在​pipeline中添加任务：在目标文件中框选召回内容并保存
    ppl.formatter = (lambda nodes, query: dict(context_str=nodes, query=query)) | bind(query=ppl.input)
    ppl.llm = lazyllm.OnlineChatModule(stream=False).prompt(lazyllm.ChatPrompter(prompt, extra_keys=["context_str"]))


if __name__ == "__main__":
    while True:
        print("✨  Welcome to your smart assistant ✨")
        
        query = input("\n🚀  Enter your query (type 'exit' to quit): \n> ")
        if query.lower() == "exit":
            print("\n👋  Exiting... Thank you for the using!")
            break

        print(f"\n✅  Received your query: {query}\n")

        answer = ppl(query)

        print("\n" + "=" * 50)
        print("🚀  ANSWER  🚀")
        print("=" * 50 + "\n")
        print(answer)
        print("\n" + "=" * 50 + "\n")