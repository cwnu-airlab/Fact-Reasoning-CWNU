import json
import torch

from transformers import ElectraForSequenceClassification, ElectraConfig, ElectraTokenizerFast
import torch.nn.functional as F


class Service:
    task = [
        {
            'name': 'QA Verification',
            'description': 'Verifying answer and supporting fact generated by QA module.'
        }
    ]

    def __init__(self):
        self.verification_model = VerificationModel()

    @classmethod
    def get_task_list(cls):
        return json.dumps(cls.task), 200

    def do(self, content):
        try:
            ret = self.verification_model.verify(content)
            if 'error' in ret.keys():
                return json.dumps(ret), 400
            return json.dumps(ret), 200
        except Exception as e:
            return json.dumps(
                {
                    'error': "{}".format(e)
                }
            ), 400


class VerificationModel(object):
    def __init__(self):
        self.device = 'cuda'

        # Initialize English Common-sense model
        config_eng = ElectraConfig.from_pretrained("google/electra-base-discriminator")
        config_eng.num_labels = 3
        self.verification_model_eng = ElectraForSequenceClassification(config=config_eng)
        self.tokenizer_eng = ElectraTokenizerFast.from_pretrained("google/electra-base-discriminator")

        # Load English trained Model
        checkpoint_eng = torch.load('0525_eng.pth', map_location=self.device)
        self.verification_model_eng.load_state_dict(checkpoint_eng['model'])
        self.verification_model_eng.to(self.device)

        # Initialize Korean Common-sense model
        config_kor = ElectraConfig.from_pretrained("monologg/koelectra-base-v3-discriminator")
        config_kor.num_labels = 3
        self.verification_model_kor_common_sense = ElectraForSequenceClassification(config=config_kor)
        self.verification_model_kor_legal = ElectraForSequenceClassification(config=config_kor)
        self.tokenizer_kor = ElectraTokenizerFast.from_pretrained("monologg/koelectra-base-v3-discriminator")

        # Load Korean trained Model
        checkpoint_kor_common_sense = torch.load('0513_kor.pth', map_location=self.device)
        checkpoint_kor_legal = torch.load('0513_kor.pth', map_location=self.device)

        self.verification_model_kor_common_sense.load_state_dict(checkpoint_kor_common_sense['model'])
        self.verification_model_kor_common_sense.to(self.device)
        self.verification_model_kor_legal.load_state_dict(checkpoint_kor_legal['model'])
        self.verification_model_kor_legal.to(self.device)

        self.verification_model = None
        self.tokenizer = None
        self.domain = None
        self.language = None

    def check_input(self, content):
        question = content.get("question", None)
        answers = content.get("answer", None)
        supporting_fact = content.get("supporting_fact", None)

        if not question or not answers or not supporting_fact:
            return {
                'error': "invalid query"
            }

        self.domain = question['domain']
        self.language = question['language']

        if self.language == 'kr':
            if self.domain == 'common-sense':
                self.verification_model = self.verification_model_kor_common_sense
            else:
                self.verification_model = self.verification_model_kor_legal
            self.tokenizer = self.tokenizer_kor
        else:
            self.verification_model = self.verification_model_eng
            self.tokenizer = self.tokenizer_eng

        return question, answers, supporting_fact

    def verify(self, content):
        question, answer, supporting_fact = self.check_input(content)

        try:
            content["label"] = self.get_label(question['text'],answer,supporting_fact)
            return content
        except Exception as e:
            return {'error': "{}".format(e)}

    def get_label(self, question, answer, supporting_fact):
        input = "{CLS} {supporting_fact} {SEP} {question} {answer}".format(
            CLS=self.tokenizer.cls_token, SEP=self.tokenizer.sep_token, supporting_fact=supporting_fact,
            question=question, answer=answer
        )

        encoded_input = self.tokenizer_eng.encode(input, return_tensors='pt')
        self.verification_model.eval()
        output = self.verification_model(encoded_input.to(self.device))
        pred = torch.argmax(F.softmax(output.logits), dim=1).item()

        if pred == 0:
            return "correct"
        else:
            return "incorrect"
