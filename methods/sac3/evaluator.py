import transformers
from transformers import AutoTokenizer
from methods.sac3 import llm_models


class Evaluate:
    def __init__(self, model):
        self.model = model
        self.prompt_temp = 'Answer the following question:\n'
        if self.model == 'falcon-7b':
            model_name = "tiiuae/falcon-7b-instruct"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.pipeline = transformers.pipeline(
                "text-generation",
                model=model_name,
                tokenizer=self.tokenizer,
                # torch_dtype=torch.bfloat16,
                # trust_remote_code=True,
                device_map="auto"
            )
        elif self.model == 'starling-7b':
            model_name = "berkeley-nest/Starling-LM-7B-alpha"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.pipeline = transformers.pipeline(
                "text-generation",
                model=model_name,
                tokenizer=self.tokenizer,
                # torch_dtype=torch.bfloat16,
                device_map="auto"
            )

    def self_evaluate(self, self_question, temperature, self_num):
        '''
        Inputs:
        self_question - original user query
        temperature - [0,1] for LLM randomness
        self_num - how many generated responses given this question

        Outputs:
        self_responses - generated responses given this question with different temperatures
        '''
        costs = 0
        self_responses = []
        prompt = self.prompt_temp + '\nQ:' + self_question

        for i in range(self_num):
            # llm model: GPTs, open-source models (falcon, guanaco)
            if self.model == "chatgpt":
                res, cost = llm_models.call_openai_model(prompt, "gpt-3.5-turbo", temperature)  # openai model call
            elif self.model == 'gpt3':
                res, cost = llm_models.call_openai_model(prompt, "gpt-3.5-turbo-instruct", temperature)
            elif self.model == 'guanaco-33b':
                res, cost = llm_models.call_guanaco_33b(prompt, max_new_tokens=200)
            elif self.model == 'falcon-7b':
                res, cost = llm_models.call_falcon_7b(self.pipeline, self.tokenizer, prompt, max_new_tokens=200)
            elif self.model == 'starling-7b':
                res, cost = llm_models.call_starling_7b(self.pipeline, self.tokenizer, prompt, temperature=1, max_new_tokens=200)
            elif self.model == 'llama':
                res, cost = llm_models.call_llama(prompt, temperature)
            # other open-sourced llms
            self_responses.append(res)
            costs += cost

        return self_responses, costs

    def perb_evaluate(self, perb_questions, temperature):
        '''
        Inputs:
        perb_questions - perturbed questions that are semantically equivalent to the original question
        temperature - [0,1] for LLM randomness

        Outputs:
        perb_responses - generated responses given the perturbed questions
        '''
        costs = 0
        perb_responses = []
        for i in range(len(perb_questions)):
            prompt = self.prompt_temp + '\nQ:' + perb_questions[i]
            # llm model: GPTs, open-source models (falcon, guanaco)
            if self.model == "chatgpt":
                res, cost = llm_models.call_openai_model(prompt, "gpt-3.5-turbo", temperature)  # openai model call
            elif self.model == 'gpt3':
                res, cost = llm_models.call_openai_model(prompt, "gpt-3.5-turbo-instruct", temperature)
            elif self.model == 'guanaco-33b':
                res, cost = llm_models.call_guanaco_33b(prompt, max_new_tokens=200)
            elif self.model == 'falcon-7b':
                res, cost = llm_models.call_falcon_7b(self.pipeline, self.tokenizer, prompt, max_new_tokens=200)
            elif self.model == 'starling-7b':
                res, cost = llm_models.call_starling_7b(self.pipeline, self.tokenizer, prompt, temperature=0.5, max_new_tokens=200)
            elif self.model == 'llama':
                res, cost = llm_models.call_llama(prompt, temperature)
            # other open-sourced llms
            perb_responses.append(res)
            costs += cost

        return perb_responses, costs