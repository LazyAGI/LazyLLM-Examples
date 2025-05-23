import lazyllm
from lazyllm import bind, _0
from lazyllm.components.formatter import decode_query_with_filepaths

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
def build_input(x, y):
   x_new = decode_query_with_filepaths(x)['query']

   return f'下面是CT医疗问诊的问题以及CT图像初步诊断。问题：{x_new}；图像初步诊断：{y}；'

def bulid_vqa():
   vqa = lazyllm.TrainableModule(base_model='Qwen2.5-VL-3B-Instruct',
               target_path='/mnt/lustre/share_data/dist/qwen2_5_vl_3b_sft_med').prompt(lazyllm.ChatPrompter(gen_prompt))

   with lazyllm.pipeline()  as vqa_ppl:
      vqa_ppl.vqa = vqa
      vqa_ppl.merge = build_input | bind(vqa_ppl.input, _0)

   with lazyllm.pipeline() as ppl:
      ppl.ifvqa = lazyllm.ifs(
         lambda x: x.startswith('<lazyllm-query>'),
         vqa_ppl,
         lambda x:x)
   return ppl

if __name__ == "__main__":
    lazyllm.WebModule(bulid_vqa(), port=range(23477, 23479)).start().wait()