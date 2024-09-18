from typing import Any

from modelsmith import Forge, OpenAIModel

from frameworks.base import BaseFramework, experiment

from tenacity import retry, stop_after_attempt


class ModelsmithFramework(BaseFramework):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.forge = Forge(
            model=OpenAIModel(self.llm_model),
            response_model=self.response_model,
            max_retries=self.retries,
        )

    def run(
        self, task: str, n_runs: int, expected_response: Any = None, inputs: dict = {}
    ) -> tuple[list[Any], float, dict, list[list[float]]]:
        @experiment(n_runs=n_runs, expected_response=expected_response, task=task)
        @retry(stop=stop_after_attempt(self.retries+1))
        def run_experiment(inputs):
            response = self.forge.generate(user_input=self.prompt.format(**inputs))
            return response

        predictions, percent_successful, metrics, latencies = run_experiment(inputs)
        return predictions, percent_successful, metrics, latencies
