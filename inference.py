import requests
import json
import re
import numpy as np
import torch
from modelscope import Model, snapshot_download, AutoModelForCausalLM, AutoTokenizer
from modelscope.models.nlp.llama2 import Llama2Tokenizer, Llama2Config
import transformers
from transformers import BitsAndBytesConfig
import tqdm


# Define global variables to store the model and the disambiguator
GLOBAL_MODEL = None
GLOBAL_TOKENIZER = None


def load_model(model_name):
    """
    Loading Models and Splitters.
    """
    global GLOBAL_MODEL, GLOBAL_TOKENIZER
    # Add loading logic for different models here
    if model_name == "Llama-2-70b-chat-ms":
        model_dir = "your_dir_Llama-2-70b-chat-ms"
        model_config = Llama2Config.from_pretrained(model_dir)
        model_config.pretraining_tp = 1
        tokenizer = Llama2Tokenizer.from_pretrained(model_dir)
        model = Model.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            config=model_config,
            device_map='auto'
        )

    elif model_name == "Mistral-7B-Instruct-v0.2":
        model = transformers.AutoModelForCausalLM.from_pretrained(
            "your_dir_Mistral-7B-Instruct-v0.2")
        tokenizer = transformers.AutoTokenizer.from_pretrained("your_dir_Mistral-7B-Instruct-v0.2")

    elif model_name == "Mixtral-8x7B-Instruct-v0.1":
        model_id = "your_dir_Mixtral-8x7B-Instruct-v0.1"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=True)

    elif model_name == "Yi-34B-Chat":
        model_path = 'your_dir_Yi-34B-Chat'
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype='auto'
        ).eval()

    elif model_name == "Llama-2-13b-chat-ms":
        model_dir = 'your_dir_Llama-2-13b-chat-ms'
        tokenizer = Llama2Tokenizer.from_pretrained(model_dir)
        model = Model.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map='auto'
        )

    elif model_name == "internlm2-chat-20b":
        model_dir = "your_dir_internlm2-chat-20b"
        tokenizer = AutoTokenizer.from_pretrained(model_dir, device_map="auto", trust_remote_code=True)
        # Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and might cause OOM Error.
        model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True,
                                                     torch_dtype=torch.float16).eval()
        # model = model.eval()

    GLOBAL_MODEL = model
    GLOBAL_TOKENIZER = tokenizer


def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def get_access_token():
    """
    Use the API Key and Secret Key to get the access_token, replacing the Application API Key and Application Secret Key in the following example
    """

    url = "your_token_rul"

    payload = json.dumps("")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json().get("access_token")


"""
This clss implements the inference of the model (including create the model).
"""


class Inference(object):
    def __init__(self, args):
        self.args = args
        self.model = args.model
        load_model(self.model)

    def predict(self, attacked_text=None):
        if self.model in ["chatgpt", "gpt4"]:
            results = self.predict_by_openai_api(self.model, attacked_text)
        if self.model in ['llama_2_70b', 'bloomz_7b1', 'llama_2_7b', 'llama_2_13b', 'qianfan_bloomz_7b_compressed',
                          'chatglm2_6b_32k', 'aquilachat_7b']:
            results = self.predict_by_qianfan_api(self.model, attacked_text)
        else:
            results = self.predict_by_local_inference(self.model, attacked_text)
        return results

    def call_openai_api(self, model, prompt):
        import openai
        from config import OPENAI_API
        openai.api_key = OPENAI_API
        if model in ['chatgpt']:
            response = openai.Completion.create(
                model="gpt-3.5-turbo-instruct",
                prompt=prompt,
                max_tokens=20,
                temperature=0
            )
            result = response['choices'][0]['text']
        else:
            response = openai.ChatCompletion.create(
                model='gpt-4-0613',
                messages=[
                    {"role": "user", "content": prompt},
                ]
            )
        result = response['choices'][0]['message']['content']
        return result


    def predict_by_openai_api(self, model, prompt):
        data_len = len(self.args.data)
        if data_len > 1000:
            data_len = 1000

        score = 0
        check_correctness = 100
        preds = []
        gts = []

        for idx in tqdm(range(data_len)):

            raw_data = self.args.data.get_content_by_idx(
                idx, self.args.dataset)
            input_text, gt = self.process_input(prompt, raw_data)

            raw_pred = self.call_openai_api(model, input_text)
            pred = self.process_pred(raw_pred)

            preds.append(pred)
            gts.append(gt)

            if check_correctness > 0:
                self.args.logger.info("gt: {}".format(gt))
                self.args.logger.info("Pred: {}".format(pred))
                self.args.logger.info("sentence: {}".format(input_text))

                check_correctness -= 1

        score = self.eval(preds, gts)
        return score

    def call_qianfan_api(self, model, text):
        if model in ['llama_2_70b', 'bloomz_7b1', 'llama_2_7b', 'llama_2_13b', 'qianfan_bloomz_7b_compressed',
                     'chatglm2_6b_32k', 'aquilachat_7b']:
            url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/" + model + "?access_token=" + get_access_token()
            payload = json.dumps({
                "messages": [
                    {
                        "role": "user",
                        "content": text
                    }
                ]
            })
            headers = {
                'Content-Type': 'application/json'
            }
            response = requests.request("POST", url, headers=headers, data=payload)
            result = response.json().get("result")
        return result

    def predict_by_qianfan_api(self, model, attacked_text):
        result_array = [0.000, 0.000, 0.000]
        score_sum = 0
        raw_pred = ""

        while True:
            raw_pred = self.call_qianfan_api(model, attacked_text)
            while raw_pred is None:
                raw_pred = self.call_qianfan_api(model, attacked_text)
            pred = raw_pred.strip().lower().replace("<pad>", "").replace("</s>", "")
            pred = pred.replace("[", "").replace("]", "")
            pred = pred.replace("+", " ").replace("-", " ")
            pred = pred.strip(",._\"\'-+=!?()&^%$#@:\\|\{\}[]<>/`\n\t\r\v\f ")

            matches = re.findall(r'(negative|neutral|positive)\s+(\d+\.\d+)', pred)
            category_dict = {match[0]: float(match[1]) for match in matches}
            categories = ['negative', 'neutral', 'positive']
            for i, category in enumerate(categories):
                result_array[i] = category_dict.get(category, 0.001)

            score_sum = sum(result_array)

            if 0.9 <= score_sum <= 1.1:
                break
            else:
                if result_array == [0, 0, 0]:
                    result_array = [0.333, 0.333, 0.333]
                result_array = softmax(result_array)
                score_sum = sum(result_array)
                if 0.9 <= score_sum <= 1.1:
                    break
                else:
                    print("error: score_sum not in [0.9, 1.1] after softmax")
                    print(f"original prediction: {raw_pred}")

        return result_array

    def predict_by_generation(self, model, attacked_text):
        global GLOBAL_MODEL, GLOBAL_TOKENIZER
        if model == "Llama-2-70b-chat-ms":
            system = "Please only output the label and the confidence score to three decimal places, in the format \"[negative]+[confidence score for negative],[neutral]+[confidence score for neutral],[positive]+[confidence score for positive]\", and nothing else. Don't write explanations nor line breaks in your replies."
            inputs = {'text': attacked_text, 'system': system, 'max_length': 512}
            output = GLOBAL_MODEL.chat(inputs, GLOBAL_TOKENIZER)
            response = output['response']
        elif model == "Mistral-7B-Instruct-v0.2":
            messages = [
                {"role": "user", "content": attacked_text}]
            encodeds = GLOBAL_TOKENIZER.apply_chat_template(messages, return_tensors="pt")
            model_inputs = encodeds.to('cuda:0')
            GLOBAL_MODEL.to('cuda:0')
            generated_ids = GLOBAL_MODEL.generate(model_inputs, max_new_tokens=1000, do_sample=True)
            decoded = GLOBAL_TOKENIZER.batch_decode(generated_ids)
            text = decoded[0]
            inst_end_index = text.rfind("[/INST]")
            s_tag_start_index = text.rfind("</s>")
            response = text[inst_end_index + len("[/INST]"):s_tag_start_index].strip()
        elif model == "Mixtral-8x7B-Instruct-v0.1":
            inputs = GLOBAL_TOKENIZER(attacked_text, return_tensors="pt")
            outputs = GLOBAL_MODEL.generate(**inputs, max_new_tokens=50)
            response = GLOBAL_TOKENIZER.decode(outputs[0], skip_special_tokens=True)
        elif model == "Yi-34B-Chat":
            messages = [{"role": "user", "content": attacked_text}]
            input_ids = GLOBAL_TOKENIZER.apply_chat_template(conversation=messages, tokenize=True,
                                                             add_generation_prompt=True,
                                                             return_tensors='pt')
            output_ids = GLOBAL_MODEL.generate(input_ids.to('cuda'))
            response = GLOBAL_TOKENIZER.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        elif model == "Llama-2-13b-chat-ms":
            system = "Please only output the label and the confidence score to three decimal places, in the format \"[negative]+[confidence score for negative],[neutral]+[confidence score for neutral],[positive]+[confidence score for positive]\", and nothing else. Don't write explanations nor line breaks in your replies."
            inputs = {'text': attacked_text, 'system': system, 'max_length': 1000}
            output = GLOBAL_MODEL.chat(inputs, GLOBAL_TOKENIZER)
            response = output['response']
        elif model == "internlm2-chat-20b":
            response, history = GLOBAL_MODEL.chat(GLOBAL_TOKENIZER, attacked_text, history=[])

        else:
            raise ValueError(f"Unsupported model: {model}. Please check the model name and try again.")
        return response

    def predict_by_local_inference(self, model, attacked_text):
        raw_pred = self.predict_by_generation(model, attacked_text)
        while raw_pred is None:
            raw_pred = self.predict_by_generation(model, attacked_text)
        pred = self.process_raw_predict(model, raw_pred)
        if np.array_equal(pred, [0.000, 0.000, 0.000]):
            pred = [0.333, 0.333, 0.333]
        return pred


    def process_raw_predict(self, model, raw_pred):
        result_array = [0.000, 0.000, 0.000]
        if model == "Llama-2-13b-chat-ms":
            text = raw_pred
            text = text.replace("\n", " ")
            text = remove_before_first_bracket(text)
            last_bracket_index = text.rfind("]")
            if last_bracket_index != -1:
                text = text[:last_bracket_index + 1]
            else:
                text = text
            text = text.strip().lower().replace("<pad>", "").replace("</s>", "")
            text = text.replace("[", "").replace("]", "")
            text = text.replace("+", " ").replace("-", " ")
            text = text.replace(":", " ").replace(",", " ")
            text = text.replace("\"", " ").replace("\'", " ")
            text = text.strip(",._\"\'-+=!?()&^%$#@:\\|\{\}[]<>/`\n\t\r\v\f ")
            matches = re.findall(r'(negative|neutral|positive)\s+(\d+\.\d+)', text)
            category_dict = {match[0]: float(match[1]) for match in matches}
            categories = ['negative', 'neutral', 'positive']
            for i, category in enumerate(categories):
                result_array[i] = category_dict.get(category, 0.001)
            score_sum = sum(result_array)
            if 0.9 <= score_sum <= 1.1:
                return result_array
            else:
                if np.array_equal(result_array, [0, 0, 0]):
                    result_array = [0.333, 0.333, 0.333]
                result_array = softmax(result_array)
                score_sum = sum(result_array)
                if 0.9 <= score_sum <= 1.1:
                    return result_array
                else:
                    print("error: score_sum not in [0.9, 1.1] after softmax")
                    print(f"original prediction: {raw_pred}")
                    result_array = [0.000, 0.000, 0.000]
                    return result_array

        elif model == "Llama-2-70b-chat-ms" or model == "internlm2-chat-20b":
            text = raw_pred
            text = text.replace("\n", " ")
            text = remove_before_first_bracket(text)
            text = text.strip().lower().replace("<pad>", "").replace("</s>", "")
            text = text.replace("[", "").replace("]", "")
            text = text.replace("+", " ").replace("-", " ")
            text = text.replace(":", " ").replace(",", " ")
            text = text.replace("\"", " ").replace("\'", " ")
            text = text.strip(",._\"\'-+=!?()&^%$#@:\\|\{\}[]<>/`\n\t\r\v\f ")
            matches = re.findall(r'(negative|neutral|positive)\s+(\d+\.\d+)', text)
            category_dict = {match[0]: float(match[1]) for match in matches}
            categories = ['negative', 'neutral', 'positive']
            for i, category in enumerate(categories):
                result_array[i] = category_dict.get(category, 0.001)
            score_sum = sum(result_array)
            if 0.9 <= score_sum <= 1.1:
                return result_array
            else:
                if np.array_equal(result_array, [0, 0, 0]):
                    result_array = [0.333, 0.333, 0.333]
                result_array = softmax(result_array)
                score_sum = sum(result_array)
                if 0.9 <= score_sum <= 1.1:
                    return result_array
                else:
                    print("error: score_sum not in [0.9, 1.1] after softmax")
                    print(f"original prediction: {raw_pred}")
                    result_array = [0.000, 0.000, 0.000]
                    return result_array

        elif model == "Mistral-7B-Instruct-v0.2":
            text = raw_pred
            text = text.split('\n')[0]
            text = re.sub(r'(?<!\d)(\.\d+)', r'0\1', text)
            text = text.replace("\n", " ")
            words = re.findall(r'[a-zA-Z]+', text)
            word_count = len(words)
            if word_count == 0:
                result_array = [0.000, 0.000, 0.000]
                return result_array
            if word_count > 4:
                result_array = [0.000, 0.000, 0.000]
                return result_array
            matches = re.findall(r'(?:\d+\.\d+,){2}\d+\.\d+', text)
            if matches:
                result_array = [0.000, 0.000, 0.000]
                return result_array
            text = remove_before_first_bracket(text)
            text = text.replace("]", "] ")
            text = text.strip().lower().replace("<pad>", "").replace("</s>", "")
            text = text.replace("[", "").replace("]", "")
            text = text.replace("+", " ").replace("-", " ")
            text = text.replace(":", " ").replace(",", " ")
            text = text.replace("\"", " ").replace("\'", " ")
            text = text.replace("\\", " ").replace("=", " ")
            text = text.strip(",._\"\'-+=!?()&^%$#@:\\|\{\}[]<>/`\n\t\r\v\f ")
            matches = re.findall(r'(negative|neutral|positive)\s+(\d+\.\d+)', text)
            category_dict = {match[0]: float(match[1]) for match in matches}
            categories = ['negative', 'neutral', 'positive']
            for i, category in enumerate(categories):
                result_array[i] = category_dict.get(category, 0.001)
            score_sum = sum(result_array)
            if 0.9 <= score_sum <= 1.1:
                return result_array
            else:
                if np.array_equal(result_array, [0, 0, 0]):
                    result_array = [0.333, 0.333, 0.333]
                result_array = softmax(result_array)
                score_sum = sum(result_array)
                if 0.9 <= score_sum <= 1.1:
                    return result_array
                else:
                    print("error: score_sum not in [0.9, 1.1] after softmax")
                    print(f"original prediction: {raw_pred}")
                    result_array = [0.000, 0.000, 0.000]
                    return result_array

        elif model == "Mixtral-8x7B-Instruct-v0.1":
            text = raw_pred
            text = text.split('\n')
            text = [line for line in text if not line.startswith("I want you")]
            text = '\n'.join(text)
            text = text.replace("\n", " ")
            contains_keywords = any(word in text for word in ["negative", "neutral", "positive"])
            contains_digit = any(char.isdigit() for char in text)
            if not (contains_keywords and contains_digit):
                result_array = [0.000, 0.000, 0.000]
                return result_array
            text = remove_before_first_bracket(text)
            text = text.replace("]", "] ")
            text = text.strip().lower().replace("<pad>", "").replace("</s>", "")
            text = text.replace("[", "").replace("]", "")
            text = text.replace("+", " ").replace("-", " ")
            text = text.replace(":", " ").replace(",", " ")
            text = text.replace("\"", " ").replace("\'", " ")
            text = text.strip(",._\"\'-+=!?()&^%$#@:\\|\{\}[]<>/`\n\t\r\v\f ")
            matches = re.findall(r'(negative|neutral|positive)\s+(\d+\.\d+)', text)
            category_dict = {match[0]: float(match[1]) for match in matches}
            categories = ['negative', 'neutral', 'positive']
            for i, category in enumerate(categories):
                result_array[i] = category_dict.get(category, 0.001)
            score_sum = sum(result_array)
            if 0.9 <= score_sum <= 1.1:
                return result_array
            else:
                if np.array_equal(result_array, [0, 0, 0]):
                    result_array = [0.333, 0.333, 0.333]
                result_array = softmax(result_array)
                score_sum = sum(result_array)
                if 0.9 <= score_sum <= 1.1:
                    return result_array
                else:
                    print("error: score_sum not in [0.9, 1.1] after softmax")
                    print(f"original prediction: {raw_pred}")
                    result_array = [0.000, 0.000, 0.000]
                    return result_array

        elif model == "Yi-34B-Chat":
            text = raw_pred
            text = text.replace("\n", " ")
            text = remove_before_first_bracket(text)
            text = text.replace("]", "] ")
            text = text.strip().lower().replace("<pad>", "").replace("</s>", "")
            text = text.replace("[", "").replace("]", "")
            text = text.replace("+", " ").replace("-", " ")
            text = text.replace(":", " ").replace(",", " ")
            text = text.replace("\"", " ").replace("\'", " ")
            text = text.strip(",._\"\'-+=!?()&^%$#@:\\|\{\}[]<>/`\n\t\r\v\f ")
            matches = re.findall(r'(negative|neutral|positive)\s+(\d+\.\d+)', text)
            category_dict = {match[0]: float(match[1]) for match in matches}
            categories = ['negative', 'neutral', 'positive']
            for i, category in enumerate(categories):
                result_array[i] = category_dict.get(category, 0.001)
            score_sum = sum(result_array)
            if 0.9 <= score_sum <= 1.1:
                return result_array
            else:
                if np.array_equal(result_array, [0, 0, 0]):
                    result_array = [0.333, 0.333, 0.333]
                result_array = softmax(result_array)
                score_sum = sum(result_array)
                if 0.9 <= score_sum <= 1.1:
                    return result_array
                else:
                    print("error: score_sum not in [0.9, 1.1] after softmax")
                    print(f"original prediction: {raw_pred}")
                    result_array = [0.000, 0.000, 0.000]
                    return result_array

        else:
            raise ValueError(f"Unsupported model: {model}. Please check the model name and try again.")


def remove_before_first_bracket(text):
    parts = text.split("[", 1)
    if len(parts) > 1:
        return "[" + parts[1]
    else:
        return text
