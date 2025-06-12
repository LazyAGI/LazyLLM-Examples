import re
from datasets import load_dataset

# 下载数据集
ds = load_dataset("FreedomIntelligence/Huatuo26M-Lite", split="train")

# 根据关键字缩减知识库，提高检索速度
pattern = re.compile(r"(头|脑|颅|心|肺|肋|胸|膈|腹|肝|胆|胰|脾|肾|胃|肠)")

# 输出文件路径
output_path = "/home/mnt/pansihan/ProjectLazyLLM/test/huatuo_data.txt"
count = 0

with open(output_path, "w", encoding="utf-8") as f:
    for sample in ds:
        question = sample.get("question", "").strip().replace("\n", " ")
        answer = sample.get("answer", "").strip().replace("\n", " ")
        diseases = sample.get("related_diseases", [])

        # 筛选逻辑：只保留包含关键字的内容
        if not pattern.search(question) and not pattern.search(answer):
            continue

        if isinstance(diseases, list):
            diseases = ", ".join(d.strip() for d in diseases)
        else:
            diseases = diseases.strip()

        if question and answer:
            line = f"Question: {question}\tAnswer: {answer}\tRelated Diseases: {diseases}"
            f.write(line + "\n")
            count += 1

print(f"✅ 数据筛选并处理完成，共写入 {count} 条记录 -> {output_path}")
