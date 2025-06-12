import lazyllm

gen_prompt = "下面是一个问题，运用医学知识来正确回答提问."

def build_inquiry():
    llm = lazyllm.TrainableModule(base_model='internlm2-chat-7b',
            target_path='/mnt/lustre/share_data/dist/internlm2-chat-7b_sft_med').prompt(lazyllm.ChatPrompter(gen_prompt))
    return llm

if __name__ == "__main__":
    lazyllm.WebModule(build_inquiry(), port=range(23471, 23473)).start().wait()