from abc import ABC, abstractmethod
from pydantic_ai import Agent

class BaseAgent(ABC):
    def __init__(self, name: str, agent: Agent):
        self._name = name
        self._agent = agent

    @abstractmethod
    def get_instruction(self):
        ...

    @abstractmethod
    def get_prompt(self):
        ...

    def get_failure_prompt(self, failure_reason: str):
        return f"""
            CORRECTION REQUIRED
            Your previous result was rejected for the following reason:
            {failure_reason}

            Return a different result that avoids this problem.
        """            
