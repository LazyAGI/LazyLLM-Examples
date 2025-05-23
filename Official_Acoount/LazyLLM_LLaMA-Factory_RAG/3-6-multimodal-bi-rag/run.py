import lazyllm
from lazyllm import OnlineChatModule, pipeline, _0
from lazyllm.tools import IntentClassifier

from statistical_agent import build_statistical_agent
# from paper_rag import build_paper_rag
from paper_rag import build_paper_rag
from common_inquiry import build_inquiry

gen_prompt = """
# 图片信息提取器
 
1. 返回格式：  
   ### 提问: [用户原问题]  
   ### 提问中涉及到的图像内容描述：[客观描述，包括主体、背景、风格等]  
2. 要求：详细、中立，避免主观猜测  

**示例：**  
输入："找类似的猫图"（附橘猫图）, 
响应如下：
   ### 提问: 找类似的猫图  
   ### 提问中涉及到的图像内容描述：一只橘猫趴在沙发上，阳光从左侧照射，背景是米色窗帘  

"""

# 构建各个工作流
paper_ppl = build_paper_rag()
sql_ppl = build_statistical_agent()
inquiry_ppl = build_inquiry()

# 搭建具备知识问答和统计问答能力的主工作流
def build_paper_assistant():
    llm = OnlineChatModule(source='qwen', stream=False)
    vqa = lazyllm.TrainableModule('Qwen2.5-VL-3B-Instruct').prompt(lazyllm.ChatPrompter(gen_prompt))

    with pipeline() as ppl:
        ppl.ifvqa = lazyllm.ifs(
            lambda x: x.startswith('<lazyllm-query>'),
            vqa, lambda x:x)
        with IntentClassifier(llm) as ppl.ic:
            ppl.ic.case["论文问答", paper_ppl]
            ppl.ic.case["统计问答", sql_ppl]

    return ppl

if __name__ == "__main__":
    main_ppl = build_paper_assistant()
    lazyllm.WebModule(main_ppl, port=23494, static_paths="./images", encode_files=True).start().wait()