import openai
import time

openai.api_key = ''


class Examiner():
    def __init__(self, claim):
        self.message_history = []
        self.claim = claim
        self.client = openai.OpenAI()

    def Setup(self):
        Prompts = 'Your goal is to try to verify the correctness of the following claim:{}, based on the background information you will gather. \
To gather this, You will provide short questions whose purpose will be to verify the correctness of the claim, and I will reply to you with the answers to these. \
Hopefully, with the help of the background questions and their answers, you will be able to reach a conclusion as to whether the claim is correct or possibly incorrect. \
Please keep asking questions as long as you are yet to be sure regarding the true veracity of the claim. Please start with the first questions.'
        message = {"role": "user", "content": Prompts.format(self.claim)}
        self.message_history.append(message)
        response = self.request_api()
        response_message = {"role": "assistant", "content": response}
        self.message_history.append(response_message)

        return response

    def check_follow_up_question(self, answer):
        Prompts = '{} Do you have any follow-up questions? Please answer with Yes or No.'
        message = {"role": "user", "content": Prompts.format(answer)}
        self.message_history.append(message)
        response = self.request_api()
        response_message = {"role": "assistant", "content": response}
        self.message_history.append(response_message)

        return response

    def decision(self):
        Prompts = 'Based on the interviewee\'s answers to your questions, what is your conclusion regarding the correctness of the claim? Do you think it is correct or incorrect? only answer with correct or incorrect.'
        message = {"role": "user", "content": Prompts}
        self.message_history.append(message)
        response = self.request_api()
        response_message = {"role": "assistant", "content": response}
        self.message_history.append(response_message)

        return self.message_history

    def ask_continue(self):
        Prompts = 'What are the follow-up questions?'
        message = {"role": "user", "content": Prompts}
        self.message_history.append(message)
        response = self.request_api()
        response_message = {"role": "assistant", "content": response}
        self.message_history.append(response_message)

        return response

    def request_api(self):
        flag = True
        while flag:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=self.message_history,
                    temperature=0,
                    max_tokens=256)

                flag = False
            except Exception:
                print("try again!")
                time.sleep(5)

        text_response = response.choices[0].message.content
        cost = response.usage.total_tokens

        return text_response


class Suspect:
    def __init__(self, query, claim):
        self.client = openai.OpenAI()
        self.message_history = []
        message = {"role": "user", "content": query}
        self.message_history.append(message)
        response_message = {"role": "assistant", "content": claim}
        self.message_history.append(response_message)

    def answer(self, question):
        Prompts = 'Please answer the following questions regarding your claim. {}'
        message = {"role": "user", "content": Prompts.format(question)}
        self.message_history.append(message)
        response = self.request_api()
        response_message = {"role": "assistant", "content": response}
        self.message_history.append(response_message)

        return response

    def answer_without_history(self, question):
        self.message_history = []
        Prompts = 'Please answer the following questions. {}'
        message = {"role": "user", "content": Prompts.format(question)}
        self.message_history.append(message)
        response = self.request_api()
        response_message = {"role": "assistant", "content": response}
        self.message_history.append(response_message)

        return response

    def request_api(self):
        flag = True
        while flag:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=self.message_history,
                    temperature=0,
                    max_tokens=256)
                flag = False
            except Exception:
                print("try again!")
                time.sleep(5)
        text_response = response.choices[0].message.content
        cost = response.usage.total_tokens

        return text_response
