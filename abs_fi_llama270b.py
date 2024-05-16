from inference import Inference
import textattack
from textattack import Attack
from textattack.goal_functions import UntargetedLLMClassification
from textattack.transformations import WordSwapWordNet
from datasets import load_from_disk
from types import SimpleNamespace
from datetime import datetime
import logging
from textattack.search_methods import BeamWordSwapWIR
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification
)


test_name = 'fi_abs_Llama-2-70b-chat-ms'
text_name = test_name + '.txt'
csv_name = test_name + '.csv'
args = SimpleNamespace(text="some text")
fi = load_from_disk('fi_dataset')
args.model = 'Llama-2-70b-chat-ms'
args.test_name = test_name
args.log_name = test_name + '.log'

logger = logging.getLogger(test_name)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(args.log_name)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
inference_model = Inference(args)
goal_function = UntargetedLLMClassification(inference=inference_model,
                                            logger=args.test_name,
                                            model_wrapper=None)
transformation = WordSwapWordNet()
constraints = [RepeatModification(), StopwordModification()]
search_method = BeamWordSwapWIR("weighted-saliency", beam_width=6, min_beam_width=1, P_max=1.0)
prompt = "I want you to act as a natural language processing model performing a text classification task. I will input the test text and you will respond with the label of the text (negative or neutral or positive) and the confidence score corresponding to each label. Please only output the label and the confidence score to three decimal places, in the format \"[negative]+[confidence score for negative],[neutral]+[confidence score for neutral],[positive]+[confidence score for positive]\", and nothing else. Don't write explanations nor line breaks in your replies. My first text is:"
attack = Attack(goal_function, constraints, transformation, search_method)
text = prompt + fi['train'][2263]['text']
label = fi['train'][2263]['label']
example_ori = [(text, label)]
for i in range(2263):
    text = prompt + fi['train'][i]['text']
    if len(text) >= 1000:
        text = text[:1000]
    label = fi['train'][i]['label']
    example_ori.append((text, label))
dataset = textattack.datasets.Dataset(example_ori)
attack_args = textattack.AttackArgs(num_examples=1000, log_to_txt=text_name, disable_stdout=True, shuffle=True,
                                    random_seed=42, log_to_csv=csv_name, parallel=False)
attacker = textattack.Attacker(attack, dataset, attack_args)
logger.info('Experiment started at ' + str(datetime.now()))
start_time = datetime.now()
attacker.attack_dataset()
logger.info('Experiment finished at ' + str(datetime.now()))
end_time = datetime.now()
duration = end_time - start_time
logger.info('Total experiment duration: ' + str(duration))

