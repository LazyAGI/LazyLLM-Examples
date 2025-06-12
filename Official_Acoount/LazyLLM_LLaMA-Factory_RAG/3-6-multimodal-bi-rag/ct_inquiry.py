import lazyllm
from lazyllm import Document,Retriever,Reranker,OnlineChatModule,pipeline,bind,TrainableModule

prompt = (
    "你是一个专业的医学问答助手，用户会提供问题和CT影像的初步诊断结果，以及用来补充参考的医学知识。"
    "请根据用户提供的信息给出相关的诊断结果和治疗建议，若用户不存在问题可以不用给出治疗建议。\n\n"
    "注意：如果资料中没有明确提到答案，也请你基出于医学常识尽力做解释。\n\n"
)

def bulid_ct_inquiry():
    documents = Document(dataset_path="./kb", embed=TrainableModule('bge-m3'))
    documents.create_node_group(name="block", transform=lambda s: s.split("\n") if s else '')

    with pipeline() as ppl:
        ppl.retriever = Retriever(doc=documents, group_name="block",
            similarity="cosine", topk=5)
        # ppl.reranker = Reranker(name='ModuleReranker',
        #                     model="bge-reranker-large",
        #                     topk=3) | bind(query=ppl.input)
        ppl.formatter = (
            lambda nodes, query: {
                "query": query, 
                "context_str": "参考：" + "".join([node.get_content() for node in nodes])}
        ) | bind(query=ppl.input)

        ppl.llm = OnlineChatModule(source="qwen", model="qwen-max-latest", stream=False).prompt(lazyllm.ChatPrompter(instruction=prompt, extra_keys=['context_str']))
    
    return ppl

if __name__ == "__main__":
    lazyllm.WebModule(bulid_ct_inquiry(), port=range(23474, 23476)).start().wait()