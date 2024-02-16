import time

import openai
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)


class Examiner():
    def __init__(self, claim, task, query=None, reference=None):
        self.message_history = []
        self.claim = claim
        self.task = task
        self.cost = 0
        self.client = openai.OpenAI()
        if task == "general_qa":
            self.additional_setup_prompt = f"\n The answer to the question: '{query}' is '{claim}' \n"
        elif task == "question_answering":
            self.additional_setup_prompt = f"\n Based on the following the knowledge: '{reference}' the answer to the following question: '{query}' is: '{claim}' \n"
        elif task == "dialogue":
            self.additional_setup_prompt = f"\n Based on the following the knowledge: '{reference}' the answer to the following last question of the conversation: '{query.split('[Human]: ')[-1]}' is: '{claim}' \n"
        elif task == "text_generation":
            self.additional_setup_prompt = f"\n '{claim}' \n"
        else:
            raise ValueError("Unknown task")

    def Setup(self):
        Prompts = "Your goal is to try to verify the correctness of the following information {}. Take your decision based on the background information you will gather. \
To gather this, You will provide short questions whose purpose will be to verify the correctness of the information. With the help of the background questions and their answers, you then should be able to reach a conclusion as to whether the information is correct or possibly incorrect. Please keep asking questions either until you detect incorrect information or as long as it takes for you to be sure that all the information is correct. When you are asked 'Do you have any follow-up questions?' please answer only Yes if you have more questions and No if you don't have any. In the next turn you can ask your question if you said yes. If none of your questions were answered, please ask different questions."
        system_prompt = "You are an examiner having a conversation with an examinee. Your goal is to try to verify the correctness of the provided information."
        logging.info(f"Prompt examiner: {Prompts.format(self.additional_setup_prompt)}")
        system_message = {"role": "system", "content": system_prompt}
        self.message_history.append(system_message)
        message = {"role": "user", "content": Prompts.format(self.additional_setup_prompt)}
        self.message_history.append(message)
        response, cost = self.request_api()
        self.cost += cost
        response_message = {"role": "assistant", "content": response}
        self.message_history.append(response_message)

        return response

    def check_follow_up_question(self, answer):
        Prompts = '{} Do you have any follow-up questions? Please answer only Yes or No.'
        message = {"role": "user", "content": Prompts.format(answer)}
        self.message_history.append(message)
        response, cost = self.request_api()
        self.cost += cost
        response_message = {"role": "assistant", "content": response}
        self.message_history.append(response_message)

        return response

    def decision(self):
        Prompts = 'Based on the interviewee\'s answers to your questions, what is your conclusion regarding the correctness of the claim? If interviewee did not answer any of your questions, categorize it as incorrect. Otherwise, judge based on the answers given. Do you think it is correct or incorrect? only answer with correct or incorrect.'
        message = {"role": "user", "content": Prompts}
        self.message_history.append(message)
        response, cost = self.request_api()
        self.cost += cost
        response_message = {"role": "assistant", "content": response}
        self.message_history.append(response_message)

        return self.message_history

    def ask_continue(self):
        Prompts = 'What are the follow-up questions?'
        message = {"role": "user", "content": Prompts}
        self.message_history.append(message)
        response, cost = self.request_api()
        self.cost += cost
        response_message = {"role": "assistant", "content": response}
        self.message_history.append(response_message)

        return response

    def request_api(self):
        flag = True
        while flag:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo-0125",
                    messages=self.message_history,
                    temperature=0,
                    max_tokens=256)

                flag = False
            except Exception:
                print("try again!")
                time.sleep(5)

        text_response = response.choices[0].message.content
        cost = response.usage.total_tokens

        return text_response, cost


class Suspect:
    def __init__(self, query, claim, task, reference):
        if task == "question_answering" or task == "dialogue":
            self.reference = reference
        else:
            self.reference = None
        self.client = openai.OpenAI()
        self.message_history = []
        message = {"role": "user", "content": query}
        self.message_history.append(message)
        response_message = {"role": "assistant", "content": claim}
        self.message_history.append(response_message)
        self.cost = 0

    def answer(self, question):
        Prompts = 'Please answer the following questions regarding your claim. {}'
        message = {"role": "user", "content": Prompts.format(question)}
        self.message_history.append(message)
        response, cost = self.request_api()
        self.cost += cost
        response_message = {"role": "assistant", "content": response}
        self.message_history.append(response_message)

        return response

    def answer_without_history(self, question):
        self.message_history = []
        if self.reference:
            Prompts = 'Please answer the following questions. {question} based on the following knowledge: {reference}'
            message = {"role": "user", "content": Prompts.format(question=question, reference=self.reference)}
        else:
            Prompts = 'Please answer the following questions. {}'
            message = {"role": "user", "content": Prompts.format(question)}
        self.message_history.append(message)
        response, cost = self.request_api()
        self.cost += cost
        response_message = {"role": "assistant", "content": response}
        self.message_history.append(response_message)

        return response

    def request_api(self):
        flag = True
        while flag:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo-0125",
                    messages=self.message_history,
                    temperature=0,
                    max_tokens=256)
                flag = False
            except Exception:
                print("try again!")
                time.sleep(5)
        text_response = response.choices[0].message.content
        cost = response.usage.total_tokens

        return text_response, cost
