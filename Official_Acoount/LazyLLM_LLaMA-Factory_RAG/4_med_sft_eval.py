"""
Huatuo-Llama-Med-Chinese Evaluation Script with Model Fine-tuning and Inference Capabilities
"""

import json
import time
import argparse
import numpy as np
from tqdm import tqdm

import lazyllm
from lazyllm import finetune, deploy, launchers
from sklearn.model_selection import train_test_split


# Template for constructing QA prompts
template = "下面是一个问题，运用医学知识来正确回答提问.\n{instruction}\n\n\n### 回答：\n"

def load_data(data_path):
    """Load JSON data from specified file path"""
    data = []
    with open(data_path) as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def save_data(data, file_path):
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def save_res(data, file_path):
    """Save data to JSON file with proper formatting"""
    with open(file_path, 'w') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def build_data(data):
    """Format training data using predefined template"""
    extracted_data = []
    for item in data:
        extracted_item = {
            "instruction": template.format(instruction=item["instruction"]),
            "input": "",
            "output": item["output"]
        }
        extracted_data.append(extracted_item)
    return extracted_data

def cosine(x, y):
    """Calculate cosine similarity between two vectors"""
    product = np.dot(x, y)
    norm = np.linalg.norm(x) * np.linalg.norm(y)
    raw_cosine = product / norm if norm != 0 else 0.0
    return max(0.0, min(raw_cosine, 1.0))

def calculate_score(eval_set, infer_set, eval_score_res_path):
    """Calculate three evaluation metrics: exact match, cosine similarity, and word containment"""
    assert len(eval_set) == len(infer_set), \
        f"The size of eval-set is {len(eval_set)}, But size of infer-res is {len(infer_set)}."

    # Initialize embedding model
    m = lazyllm.TrainableModule('bge-large-zh-v1.5')
    m.start()

    accu_cosin_score = 0
    res = []
    for index, eval_item in enumerate(eval_set):
        output = infer_set[index].strip()
        true_answer = eval_item['output']

        # Cosine similarity scoring:
        outputs = json.loads(m([output, true_answer]))
        cosine_score = cosine(outputs[0], outputs[1])
        accu_cosin_score += cosine_score

        res.append({'question': eval_item['instruction'],
                    'true_answer': true_answer,
                    'prediction': output,
                    'cosine_score': cosine_score})
    m.stop()
    save_res(res, eval_score_res_path)
    total_score = len(eval_set)
    return (f'Cosine Score: {accu_cosin_score}/{total_score}, {round(accu_cosin_score/total_score,4)*100}%\n')

def online_infer(model, data):
    res_list = []
    for x in tqdm(data, desc="Processing"):
        try_times = 1
        while try_times < 5:
            try:
                res = model(x)
                if res:
                    try_times = 10
                    res_list.append(res)
                else:
                    try_times += 1
            except Exception:
                try_times += 1
        if try_times != 10:
            res_list.append('')
    return res_list

def main(model_path, mode, data_path, test_output_path, train_output_path, eval_score_res_path, eval_res_path):
    """Main execution flow for different operation modes"""
    # Load data
    data = load_data(data_path)
    extracted_data = build_data(data)
    train_data, test_data = train_test_split(extracted_data, test_size=0.1, random_state=42)
    save_data(train_data, train_output_path)
    save_data(test_data, test_output_path)
    eval_data = [item["instruction"] for item in test_data]

    # Online inference mode
    if mode == 'online_infer':
        model = lazyllm.OnlineChatModule(model_path)
        eval_res = online_infer(model, eval_data)
        # eval_res = [model(x) for x in tqdm(eval_data, desc="Processing")]

    # Local model operations
    if mode in ('local_infer', 'local_train'):
        model = lazyllm.TrainableModule(model_path)\
            .mode('finetune')\
            .trainset(train_output_path)\
            .finetune_method((finetune.llamafactory, {
                'learning_rate': 1e-4,
                'cutoff_len': 5120,
                'max_samples': 20000,
                'val_size': 0.05,
                'per_device_train_batch_size': 2,
                'num_train_epochs': 2.0,
                'launcher': launchers.sco(nnode=1, nproc=1, ngpus=8)
            }))\
            .prompt(dict(system='You are a helpful assistant.', drop_builtin_system=True))\
            .deploy_method(deploy.Vllm)
        model.evalset(eval_data)
        if mode == 'local_train':
            model.update()  # Auto: Start fine-tuning -> Launch inference service -> Run evaluation
        else:
            model.start()  # Start inference service
            model.eval()  # Run evaluation
        eval_res = model.eval_result
    # Score calculation mode
    if mode == 'score':
        infer_res = load_data(eval_res_path)
        eval_res = [item['infer'] for item in infer_res]

    # Calculate and display final scores
    score = calculate_score(test_data, eval_res, eval_score_res_path)
    time.sleep(5)  # Buffer for log synchronization
    print("All Done. Score is: ", score)

if __name__ == '__main__':
    # Command-line argument configuration
    parser = argparse.ArgumentParser(description="Model Training and Evaluation Pipeline")
    parser.add_argument('--model_path', type=str, default='internlm2-chat-7b',
                        help='Path to model or model identifier')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Custom path to all data')
    parser.add_argument('--train_output_path', type=str, default=None,
                        help='Custom path to training data')
    parser.add_argument('--test_output_path', type=str, default=None,
                        help='Custom path to evaluation data')
    parser.add_argument('--eval_res_path', type=str, default=None,
                        help='Path to pre-computed inference results')
    parser.add_argument('--eval_score_res_path', type=str, default=None,
                        help='Path to evaluation results')
    parser.add_argument('--mode', type=str, default='local_infer',
                        choices=['online_infer', 'local_infer', 'local_train', 'score'],
                        help='Operation mode selection')
    args = parser.parse_args()

    # Execute main pipeline
    main(args.model_path, args.mode, args.data_path, args.test_output_path,
         args.train_output_path, args.eval_score_res_path, args.eval_res_path)
