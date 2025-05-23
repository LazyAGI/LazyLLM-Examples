import lazyllm
from lazyllm.tools import IntentClassifier
classifier_llm = lazyllm.OnlineChatModule(source="qwen")
chatflow_intent_list = ["论文问答", "统计问答", "普通医疗问诊","CT医疗问诊"]
classifier = IntentClassifier(classifier_llm, intent_list=chatflow_intent_list)
classifier.start()

print(classifier('论文中的weighted ILI 是什么？'))
# 输出 >>> 论文问答
print(classifier('查询数据库中有多少篇论文'))
# 输出 >>> 统计问答
print(classifier('医生，我最近总是头晕，是怎么回事？'))
# 输出 >>> 医疗问诊
print(classifier('医生，我最近做了一次胸部 CT，报告上写着‘右肺上叶磨玻璃结节’，这个是什么意思？严重吗？需要进一步检查或治疗吗？'))
# 输出 >>> CT医疗问诊