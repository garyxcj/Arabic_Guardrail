import torch
import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Tool
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score


def check_response(response, target):
    response = response.split("</think>")[-1].strip()
    if response.lower().startswith(target.lower()):
        return True
    elif f' {target.lower()}' in response.lower() or f'\n{target.lower()}' in response.lower():
        return True
    else:
        return False


class SafetyCheckerTool(Tool):
    name = "safety_checker"
    description = (
        "This is a tool that formulates the final answer correctly. It returns 1 if the judgement is unsafe, else returns 0."
    )
    inputs = {"judgement": {"type": "string", "description": "The final safety assesment"}}
    output_type = "integer"

    def forward(self, judgement, keywords=None):
        if keywords is None:
            keywords = {0: 'safe', 1: 'unsafe'}
        if check_response(judgement, keywords[0]) and check_response(judgement, keywords[1]):
            return 1
        elif check_response(judgement, keywords[0]):
            return 0
        elif check_response(judgement, keywords[1]):
            return 1
        else:
            return 0


class LocalLLMEngine:
    def __init__(self, api_key=None, base_url=None, model="meta-llama/Llama-3.1-8B-Instruct", cache_dir="hf_cache", thinking=False):
        self.model_id = model
        self.device = "cuda"
        dtype = torch.bfloat16
        print("loading model {}".format(self.model_id))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=dtype, device_map=self.device)
        self.thinking = thinking

    def get_response_message(self, messages, model=None, max_completion_tokens=1000):
        assert model is None

        if self.model_id in ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-8B"]:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.thinking  # Switches between thinking and non-thinking modes. Default is True.
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

            # conduct text completion
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_completion_tokens, temperature=0, do_sample=False
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

            # parsing thinking content
            try:
                # rindex finding 151668 (</think>)
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0

            # thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            response = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        else:
            input_ids = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(self.device)
            prompt_len = input_ids.shape[-1]
            outputs = self.model.generate(input_ids, max_new_tokens=max_completion_tokens, temperature=0, do_sample=False)
            response = self.tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)

        message = {
            "role": "assistant",
            "content": response,
        }
        return message


class GuardModel:
    def __init__(self, backend='llama3_1', debug=False):
        self.debug = debug
        if backend in ['llama3_1', 'llama3', 'qwen3_600m', 'qwen3_600m_think', 'qwen3_8b']:
            if backend == 'llama3_1':
                local_model_id = "meta-llama/Llama-3.1-8B-Instruct"
            elif backend == 'llama3':
                local_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
            elif backend == 'llama3_2_3b':
                local_model_id = "meta-llama/Llama-3.2-3B-Instruct"
            elif backend == 'qwen3_600m' or backend == 'qwen3_600m_think':
                local_model_id = "Qwen/Qwen3-0.6B"
            elif backend == 'qwen3_8b':
                local_model_id = "Qwen/Qwen3-8B"
            else:
                raise Exception("Unknown backend: {}".format(backend))
            thinking = '_think' in backend
            self.llm_engine = LocalLLMEngine(model=local_model_id, thinking=thinking)
        else:
            raise NotImplementedError(backend)
        self.safety_checker = SafetyCheckerTool()

        with open('prompt.txt', 'r') as file:
            self.prompt = file.read()

    def inference(self, text):
        if self.debug:
            print(text)

        current_prompt = self.prompt.replace("{prompt}", text)
        messages = [{"role": "user", "content": current_prompt}]
        response = self.llm_engine.get_response_message(messages=messages)
        if self.debug:
            print(response)
        return self.safety_checker(response['content'])


def load_dataset(dataset='openaimod'):
    if dataset == 'openaimod':
        instances, categories = [], []
        with open('openaimod.jsonl') as f:
            lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            instance = data['prompt']
            instances.append(instance)
            category = 0
            for key in data.keys():
                if key != 'prompt' and data[key] != 0:
                    category = 1
                    break
            categories.append(category)
    elif dataset == 'twinsafety':
        instances, categories = [], []
        with open('twinsafety.jsonl') as f:
            lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            instance = data['prompt']
            instances.append(instance)
            category = 1 if 'unsafe' in data['label'].lower() else 0
            categories.append(category)
    else:
        raise Exception("Unknown dataset: {}".format(dataset))
    return instances, categories

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama3_1", help="model")
    parser.add_argument("--dataset", type=str, default='openaimod', choices=['openaimod', 'twinsafety'], help="dataset")
    args = parser.parse_args()

    model = GuardModel(backend=args.model)
    prompts, labels = load_dataset(dataset=args.dataset)

    predictions = []
    for prompt in tqdm(prompts):
        pred = model.inference(text=prompt)
        predictions.append(pred)

    # Calculate and print metrics
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    print(f"Dataset: {args.dataset}, Accuracy: {accuracy * 100:.2f}, F1: {f1 * 100:.2f}")
