from typing import Any

from openai import OpenAI

from frameworks.base import BaseFramework, experiment

from tenacity import retry, stop_after_attempt


class VanillaOpenAIFramework(BaseFramework):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.openai_client = OpenAI()

    def run(
        self, task: str, n_runs: int, expected_response: Any = None, inputs: dict = {}
    ) -> tuple[list[Any], float, dict, list[list[float]]]:
        @experiment(n_runs=n_runs, expected_response=expected_response, task=task)
        @retry(stop=stop_after_attempt(self.retries+1))
        def run_experiment(inputs):
            response = self.openai_client.beta.chat.completions.parse(
                model=self.llm_model,
                response_format=self.response_model,
                messages=[
                    {"role": "user", "content": self.prompt.format(**inputs)}
                ],
            )
            return response.choices[0].message.parsed

        predictions, percent_successful, metrics, latencies = run_experiment(inputs)
        return predictions, percent_successful, metrics, latencies
