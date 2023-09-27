# Import necessary libraries
import together  # Import the 'together' library
import os  # Import the 'os' library for environment variable access
from typing import Any, Dict  # Import types for type hinting
from pydantic import Extra, root_validator  # Import pydantic for data validation
from langchain.llms.base import LLM  # Import a base language model class
from langchain.utils import get_from_dict_or_env  # Import a utility function
from dotenv import load_dotenv  # Import 'load_dotenv' to load environment variables

# Load environment variables from a .env file
load_dotenv()

# Set your API key from the environment variables
together.api_key = os.environ["TOGETHER_API_KEY"]

# Start the Together model with a specific endpoint
together.Models.start("togethercomputer/llama-2-70b-chat")


# Create a class to define the behavior of the Together Large Language Model (LLM)
class TogetherLLM(LLM):
    """Together large language models."""

    # Define class-level variables for configuration
    model: str = "togethercomputer/llama-2-70b-chat"  # The model endpoint to use
    together_api_key: str = os.environ["TOGETHER_API_KEY"]  # Together API key
    temperature: float = 0.7  # Sampling temperature to use
    max_tokens: int = 512  # Maximum number of tokens to generate in the completion

    class Config:
        extra = Extra.forbid  # Disallow extra fields in the configuration

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the API key is set."""
        # Get the API key from either the provided values or environment variables
        api_key = get_from_dict_or_env(values, "together_api_key", "TOGETHER_API_KEY")
        values["together_api_key"] = api_key
        return values

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "together"

    def _call(
            self,
            prompt: str,
            **kwargs: Any,
    ) -> str:
        """Call to Together endpoint."""
        # Set the Together API key for this instance
        together.api_key = self.together_api_key

        # Generate text using the Together API
        output = together.Complete.create(
            prompt,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        # Extract the generated text from the API response
        text = output['output']['choices'][0]['text']
        return text
