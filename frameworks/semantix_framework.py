from typing import Any, Optional
from dataclasses import dataclass

from semantix import enhance
from semantix.llms import OpenAI
from semantix.utils import create_class, create_enum

from frameworks.base import BaseFramework, experiment

llm = OpenAI(model="gpt-4o-mini-2024-07-18", max_tokens=2048)

## Multilabel classification task

multilabel_classes = ['lists_createoradd', 'calendar_query', 'email_sendemail', 'news_query',
 'play_music', 'play_radio', 'qa_maths', 'email_query','weather_query',
 'calendar_set','iot_hue_lightdim','takeaway_query','social_post'
 'email_querycontact','qa_factoid','calendar_remove','cooking_recipe',
 'lists_query','general_quirky','alarm_query','takeaway_order',
 'iot_hue_lightup','lists_remove','qa_currency','play_game',
 'play_audiobook','qa_definition','music_query','datetime_query',
 'transport_query','iot_hue_lightoff','iot_hue_lightchange',
 'iot_hue_lighton','alarm_set','music_likeness','recommendation_movies',
 'transport_ticket','recommendation_locations','audio_volume_mute',
 'iot_wemo_on','play_podcasts','datetime_convert','audio_volume_other',
 'recommendation_events','alarm_remove','iot_coffee','music_dislikeness',
 'general_joke','social_query']

Label = create_enum("Label", {name: name for name in multilabel_classes}, "Multilabel Classes")

@enhance("Classify the given text", llm, retries=2, method="Reason")
def classify(text: str) -> list[Label]: ... # type: ignore

## Named Entity Recognition task

ner_entities = ['passport_number', 'bank_routing_number', 'account_pin', 'swift_bic_code', 'password', 
                'credit_card_number', 'email', 'phone_number', 'person_name', 'iban', 'ipv6', 'api_key', 
                'street_address', 'company', 'local_latlng', 'time', 'employee_id', 'customer_id', 'date_of_birth', 
                'ipv4', 'bban']

NER = create_class("NER", {name: (Optional[list[str]], None)  for name in ner_entities}, "Named Entities present in the text")

@enhance("Extract named entities from the given text", llm, retries=2, method="Reason")
def extract_entities(text: str) -> NER: ...  # type: ignore

## Synthetic data generation task

@dataclass
class UserAddress:
    street: str
    city: str
    six_digit_postal_code: int
    country: str

@dataclass
class User:
    name: str
    age: int
    address: UserAddress

@enhance("Generate a random person's information. The name must be chosen at random. Make it something you wouldn't normally choose.", llm, retries=2, method="Reason")
def generate_user_data() -> User: ... # type: ignore

class SemantixFramework(BaseFramework):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def run(
        self, task: str, n_runs: int, expected_response: Any = None, inputs: dict = {}
    ) -> tuple[list[Any], float, dict, list[list[float]]]:

        @experiment(n_runs=n_runs, expected_response=expected_response, task=task)
        def run_experiment(task, **kwargs):
            if task == "multilabel_classification":
                return classify(**kwargs)
            elif task == "ner":
                return extract_entities(**kwargs)
            elif task == "synthetic_data_generation":
                return generate_user_data()
            else:
                raise ValueError(f"{task} is not allowed. Allowed values are ['multilabel_classification', 'ner', 'synthetic_data_generation']")

        predictions, percent_successful, metrics, latencies = run_experiment(task, **inputs)

        return predictions, percent_successful, metrics, latencies

