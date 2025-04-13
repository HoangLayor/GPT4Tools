import os
import pandas as pd
from glob import glob
import PIL
from tqdm import tqdm
import json
from dotenv import load_dotenv

from chat_owlviz import ConversationBot
from scripts.scorer import question_scorer

load_dotenv()
cache_dir = os.getenv("CACHE_DIR")

load_dict = {'Text2Box': 'cuda:0', 'Segmenting': 'cuda:0', 'Inpainting': 'cuda:0', 'ImageCaptioning': 'cuda:0', 'VisualQuestionAnswering': 'cuda:0', 'Image2Pose': 'cpu'}
# load_dict = {'ImageCaptioning': 'cuda:0'}
llm_kwargs = {'base_model': os.getenv("BASE_MODEL"),
                'lora_model': os.getenv("LORA_MODEL"),
                'device': "cuda:0",
                'temperature': 0.1,
                'max_new_tokens': 1024,
                'top_p': 0.75,
                'top_k': 40,
                'num_beams': 1,
                'cache_dir': os.getenv("CACHE_DIR"),}
bot = ConversationBot(load_dict=load_dict, llm_kwargs=llm_kwargs)
bot.init_agent(lang="English")


def make_dataset():
    data_dir = os.getenv("DATA_DIR")
    def make_file_path(row):
        return f"{data_dir}/images/" + row["file_name"]

    df = pd.read_csv(f"{data_dir}/usable_data.csv")
    df["file_name"] = df.apply(make_file_path, axis=1)
    return df

PROMPT = """Look at the image and answer the question.

# Answer format
Your final answer should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
- If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise.
- If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise.
- If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.

Question: {{question}}
Your final answer: """

dataset = make_dataset()

result = {"statistics": {}, "results": []}
total = len(dataset)
correct = 0

start_idx = 0
# begin from start_idx
for row_idx, row in tqdm(dataset.iloc[start_idx:].iterrows(), total=len(dataset)-start_idx):
    question = row["instruction"]
    ans = row["true_answer"]
    prompt = PROMPT.replace("{{question}}", question)
    # res = llm(image, prompt)
    
    image = row["file_name"]
    print(image)
    # res = llm(image, prompt)
    res = bot.run_query(prompt, image)
    try:
        # image = resize(PIL.Image.open(row["file_name"]), input_size)
        image = row["file_name"]
        # res = llm(image, prompt)
        res = bot.run_query(prompt, image)
    except Exception as e:
        res = ""
        print(f"Lỗi xảy ra: {e}")
        print("*"*50)

    print(f"- id: {row_idx}\n- question: {question}")
    print(f"- true answer: {ans} - prediction: {res} > {question_scorer(ans, res)}")
    result["results"].append({
        "id": row_idx,
        "question": question,
        "file_path": row["file_name"],
        "true_answer": ans,
        "prediction": res,
        "exact_match": question_scorer(ans, res),
    })
    correct += question_scorer(ans, res)

result["statistics"]["accuracy"] = correct / total
print(f"Accuracy : {correct / total}")

model = "vicuna-7b-v1.5-gpt4tools"
save_path = f"./results/{model}-results.json"

print(f"Save path: {save_path}")

with open(save_path, "w") as f:
    json.dump(result, f, indent=4)