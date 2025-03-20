from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch
import logging
logger = logging.getLogger("main")
from transformers import StoppingCriteria, StoppingCriteriaList

class StoppingCriteriaSub(StoppingCriteria):
    
    def __init__(self, stops=[]):
        StoppingCriteria.__init__(self),
        self.stops = stops
        self.length = len(stops)
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, stops=[]):
        exit = True
        for i in range(1, self.length, 1):
            if input_ids[0][-i] != self.stops[-i]:
                exit = False
        return exit
    

def initialize_model_tokenizer(pipeline_params, device):
    model = AutoModelForCausalLM.from_pretrained(pipeline_params['model_name'], torch_dtype=torch.float16).to(device)

    tokenizer = AutoTokenizer.from_pretrained(pipeline_params['tokenizer_name'], padding_side="left")

    tokenizer.pad_token = tokenizer.eos_token
    logger.info(f'Model {model} and Tokenizer {tokenizer} initialized.')

    contriever = AutoModel.from_pretrained("facebook/contriever-msmarco").to(device)
    contriever_tokenizer = AutoTokenizer.from_pretrained("facebook/contriever-msmarco")
    logger.info(f'Contriever {contriever} and Tokenizer {contriever_tokenizer} initialized.')

    stopping_criteria_dict = MODEL_to_SC[pipeline_params['model_name']]
    

    return model, tokenizer, contriever, contriever_tokenizer, stopping_criteria_dict


# SC stands for stopping criteria.
MODEL_to_SC = {
    "lmsys/vicuna-7b-v1.5": {
        "facts": StoppingCriteriaList([StoppingCriteriaSub(stops=[8015, 2546, 1490, 2114, 29901])]),
        "subq": StoppingCriteriaList([StoppingCriteriaSub(stops=[13, 4035, 12470, 29901])]),
        "done": StoppingCriteriaList([StoppingCriteriaSub(stops=[25632, 29889])]),
        "end_block": StoppingCriteriaList([StoppingCriteriaSub(stops=[2023, 4515, 1996, 3796])])
    },
    "mistralai/Mistral-7B-Instruct-v0.2": {
        "facts": StoppingCriteriaList([StoppingCriteriaSub(stops=[8637, 10212, 286, 1639, 28747])]),
        "subq": StoppingCriteriaList([StoppingCriteriaSub(stops=[5078, 17496, 28747])]),
        "done": StoppingCriteriaList([StoppingCriteriaSub(stops=[384, 538, 28723])]),
        "end_block": StoppingCriteriaList([StoppingCriteriaSub(stops=[851, 9675, 272, 2724, 28723])])
    },
    "meta-llama/Meta-Llama-3-8B": {
        "facts": StoppingCriteriaList([StoppingCriteriaSub(stops=[12289, 83712, 2144, 25])]),
        "subq": StoppingCriteriaList([StoppingCriteriaSub(stops=[3214, 7998, 25])]),
        "done": StoppingCriteriaList([StoppingCriteriaSub(stops=[17911, 13])]),
        "end_block": StoppingCriteriaList([StoppingCriteriaSub(stops=[2028, 10548, 279, 2565, 13])])
    },
    "meta-llama/Meta-Llama-3.1-8B": {
        "facts": StoppingCriteriaList([StoppingCriteriaSub(stops=[12289, 83712, 2144, 25])]),
        "subq": StoppingCriteriaList([StoppingCriteriaSub(stops=[3214, 7998, 25])]),
        "done": StoppingCriteriaList([StoppingCriteriaSub(stops=[17911, 13])]),
        "end_block": StoppingCriteriaList([StoppingCriteriaSub(stops=[2028, 10548, 279, 2565, 13])])
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "facts": StoppingCriteriaList([StoppingCriteriaSub(stops=[12020, 82612, 2097, 25])]),
        "subq": StoppingCriteriaList([StoppingCriteriaSub(stops=[3136, 7841, 25])]),
        "done": StoppingCriteriaList([StoppingCriteriaSub(stops=[17453, 13])]),
        "end_block": StoppingCriteriaList([StoppingCriteriaSub(stops=[1986, 10335, 279, 2504, 25])])
    }
}

REL2SUBQ = {
    '{} is a citizen of': 'What is the country of citizenship of {}?',
    'The author of {} is': 'Who is the author of {}?',
    'The capital of {} is': 'What is the capital of {}?',
    '{} is located in the continent of': 'Which continent is {} located in?',
    '{} was created by': 'Who created {}?',
    '{} was born in the city of': 'Which city was {} born in?',
    '{} is associated with the sport of': 'Which sport is {} associated with?',
    '{} was created in the country of': 'Which country was {} created in?',
    'The official language of {} is': 'What is the official language of {}?',
    '{} is married to': 'Who is {} married to?',
    '{} was founded by': 'Who founded {}?',
    '{} plays the position of': 'What position does {} play?',
    'The name of the current head of the {} government is': 'Who is the current head of the {} government?',
    'The company that produced {} is': 'Which company produced {}?',
    '{} is affiliated with the religion of': 'Which religion is {} affiliated with?',
    '{} was developed by': 'Who developed {}?',
    'The headquarters of {} is located in the city of': 'Which city is the headquarters of {} located in?',
    '{} was founded in the city of': 'Which city was {} founded in?',
    'The origianl broadcaster of {} is': 'Who is the original broadcaster of {}?',
    '{} is employed by': 'Who employs {}?',
    '{} speaks the language of': 'Which language does {} speak?',
    'The name of the current head of state in {} is': 'Who is the current head of state in {}?',
    '{} died in the city of': 'Which city did {} die in?',
    'The chairperson of {} is': 'Who is the chairperson of {}?',
    '{} was performed by': 'Who performed {}?',
    'The type of music that {} plays is': 'What type of music does {} play?',
    "{}'s child is": 'Who is the child of {}?',
    'The {} is': 'What is the {}?',
    '{} works in the field of': 'Which field does {} work in?',
    '{} worked in the city of': 'Which city did {} work in?',
    'The head coach of {} is': 'Who is the head coach of {}?',
    'The univeristy where {} was educated is': 'Which university was {} educated at?',
    'The director of {} is': 'Who is the director of {}?',
    '{} is famous for': 'What is {} famous for?',
    'The chief executive officer of {} is': 'Who is the chief executive officer of {}?',
    '{} was written in the language of': 'Which language was {} written in?',
    'The origianl language of {} is': 'What is the origianl language of {}?',
    'The original language of {} is': 'What is the original language of {}?'

}