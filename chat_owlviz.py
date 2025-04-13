# coding: utf-8
import os
import re
import json
import uuid
from PIL import Image, ImageDraw, ImageOps, ImageFont
import numpy as np
import argparse
import inspect
from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from gpt4tools.llm import LlamaLangChain
from gpt4tools.tools import *


GPT4TOOLS_PREFIX = """GPT4Tools can handle various text and visual tasks, such as answering questions and providing in-depth explanations and discussions. It generates human-like text and uses tools to indirectly understand images. When referring to images, GPT4Tools follows strict file name rules. To complete visual tasks, GPT4Tools uses tools and stays loyal to observation outputs. Users can provide new images to GPT4Tools with a description, but tools must be used for subsequent tasks.
TOOLS:
------

GPT4Tools has access to the following tools:"""

GPT4TOOLS_FORMAT_INSTRUCTIONS = """To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
{ai_prefix}: [your response here]
```
"""

GPT4TOOLS_SUFFIX = """Follow file name rules and do not fake non-existent file names. Remember to provide the image file name loyally from the last tool observation.

Previous conversation:
{chat_history}

New input: {input}
GPT4Tools needs to use tools to observe images, not directly imagine them. Thoughts and observations in the conversation are only visible to GPT4Tools. When answering human questions, repeat important information. Let's think step by step.
{agent_scratchpad}"""


os.makedirs('image', exist_ok=True)


def cut_dialogue_history(history_memory, keep_last_n_paragraphs=1):
    if history_memory is None or len(history_memory) == 0:
        return history_memory
    paragraphs = history_memory.split('Human:')
    if len(paragraphs) <= keep_last_n_paragraphs:
        return history_memory
    return 'Human:' + 'Human:'.join(paragraphs[-1:])

class ConversationBot:
    def __init__(self, load_dict, llm_kwargs):
        # load_dict = {'VisualQuestionAnswering':'cuda:0', 'ImageCaptioning':'cuda:1',...}
        print(f"Initializing GPT4Tools, load_dict={load_dict}")
        if 'ImageCaptioning' not in load_dict:
            raise ValueError("You have to load ImageCaptioning as a basic function for GPT4Tools")

        self.models = {}
        # Load Basic Foundation Models
        for class_name, device in load_dict.items():
            self.models[class_name] = globals()[class_name](device=device)

        # Load Template Foundation Models
        for class_name, module in globals().items():
            if getattr(module, 'template_model', False):
                template_required_names = {k for k in inspect.signature(module.__init__).parameters.keys() if k!='self'}
                loaded_names = set([type(e).__name__ for e in self.models.values()])
                if template_required_names.issubset(loaded_names):
                    self.models[class_name] = globals()[class_name](
                        **{name: self.models[name] for name in template_required_names})
        
        print(f"All the Available Functions: {self.models}")

        self.tools = []
        for instance in self.models.values():
            for e in dir(instance):
                if e.startswith('inference'):
                    func = getattr(instance, e)
                    self.tools.append(Tool(name=func.name, description=func.description, func=func))
        self.llm = LlamaLangChain(model_kwargs=llm_kwargs) 
        self.memory = ConversationBufferMemory(memory_key="chat_history", output_key='output')

    def init_agent(self, lang):
        self.memory.clear() #clear previous history
        if lang=='English':
            PREFIX, FORMAT_INSTRUCTIONS, SUFFIX = GPT4TOOLS_PREFIX, GPT4TOOLS_FORMAT_INSTRUCTIONS, GPT4TOOLS_SUFFIX
        else:
            raise NotImplementedError(f'{lang} is not supported yet')
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent="conversational-react-description",
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=True,
            agent_kwargs={'prefix': PREFIX, 'format_instructions': FORMAT_INSTRUCTIONS,
                          'suffix': SUFFIX}, )

    def run_text(self, text, state, temperature, top_p, max_new_tokens, keep_last_n_paragraphs):
        self.llm.set_llm_params(temperature=temperature,
                                top_p=top_p,
                                max_new_tokens=max_new_tokens)
        self.agent.memory.buffer = cut_dialogue_history(self.agent.memory.buffer, keep_last_n_paragraphs)
        res = self.agent({"input": text.strip()})
        res['output'] = res['output'].replace("\\", "/")
        response = re.sub('(image/[-\w]*.png)', lambda m: f'![](file={m.group(0)})*{m.group(0)}*', res['output'])
        state = state + [(text, response)]
        print(f"\nProcessed run_text, Input text: {text}\nCurrent state: {state}\n"
              f"Current Memory: {self.agent.memory.buffer}")
        image_filenames = re.findall('image/.*.png', str(self.agent.memory.buffer))
        image_filename = image_filenames[-1] if len(image_filenames) > 0 else ''
        return state, f'{image_filename} '

    def run_image(self, image, state, lang='English'):
        if image is None:
            return state
        image_filename = os.path.join('image', f"{str(uuid.uuid4())[:8]}.png")
        if not os.path.exists(image):
            return state
        
        image = Image.open(image)
        print("======>Auto Resize Image...")
        img = image
        width, height = img.size
        ratio = min(512 / width, 512 / height)
        width_new, height_new = (round(width * ratio), round(height * ratio))
        width_new = int(np.round(width_new / 64.0)) * 64
        height_new = int(np.round(height_new / 64.0)) * 64
        img = img.resize((width_new, height_new))
        img = img.convert('RGB')
        img.save(image_filename, "PNG")
        print(f"Resize image form {width}x{height} to {width_new}x{height_new}")
        description = self.models['ImageCaptioning'].inference(image_filename)
        if lang == 'English':
            Human_prompt = f'\nHuman: Provide an image named {image_filename}. The description is: {description}. Understand the image using tools.\n'
            AI_prompt = "Received."
        else:
            raise NotImplementedError(f'{lang} is not supported yet')
        
        self.agent.memory.buffer = self.agent.memory.buffer + Human_prompt + 'AI: ' + AI_prompt
        state = state + [(f"![](file={image_filename})*{image_filename}*", AI_prompt)]
        print(f"\nProcessed run_image, Input image: {image_filename}\nCurrent state: {state}\n"
              f"Current Memory: {self.agent.memory.buffer}")
        return state, f'{image_filename} '

    def run_query(self, text=None, image_path=None):
        state = []
        if text:
            state, file_name = self.run_text(text, state, 0.1, 0.75, 1024, 1)
        if image_path:
            state, file_name = self.run_image(image_path, state,)

        return state[0][1] if state else "No input provided"

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--base_model', type=str, required=True, help='folder path to the vicuna with tokenizer')
    # parser.add_argument('--lora_model', type=str, required=True, help='folder path to the lora model')
    # parser.add_argument('--load', type=str, default='ImageCaptioning_cuda:0,Text2Image_cuda:0')
    # parser.add_argument('--llm_device', type=str, default='cpu', help='device to run the llm model')
    # parser.add_argument('--temperature', type=float, default=0.1, help='temperature for the llm model')
    # parser.add_argument('--max_new_tokens', type=int, default=1024, help='max number of new tokens to generate')
    # parser.add_argument('--top_p', type=float, default=0.75, help='top_p for the llm model')
    # parser.add_argument('--top_k', type=int, default=40, help='top_k for the llm model')
    # parser.add_argument('--num_beams', type=int, default=1, help='num_beams for the llm model')
    # parser.add_argument('--keep_last_n_paragraphs', type=int, default=1, help='keep last n paragraphs in the memory')
    # parser.add_argument('--cache-dir', type=str, default=None, help="cache path to save model")
    # parser.add_argument('--server-name', type=str, default='0.0.0.0', help="gradio sever name")
    # parser.add_argument('--server-port', type=int, default=8888, help="gradio server port")
    # parser.add_argument('--share', action="store_true")
    # args = parser.parse_args()

    # load_dict = {e.split('_')[0].strip(): e.split('_')[1].strip() for e in args.load.split(',')}
    load_dict = {'ImageCaptioning': 'cuda:0'}
    # llm_kwargs = {'base_model': args.base_model,
    #               'lora_model': args.lora_model,
    #               'device': args.llm_device,
    #               'temperature': args.temperature,
    #               'max_new_tokens': args.max_new_tokens,
    #               'top_p': args.top_p,
    #               'top_k': args.top_k,
    #               'num_beams': args.num_beams,
    #               'cache_dir': args.cache_dir,}
    
    import os
    from dotenv import load_dotenv

    load_dotenv()
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
    bot.run_query(text="Is anyone other than the seller looking at the kid in this photo? Provide a yes/no answer", image_path="../data/images/market.jpeg")
    bot.run_query(text="What is the color of the top of the bus's roof? Provide a one-word answer.", image_path="../data/images/findbus.jpg")
    bot.memory.clear