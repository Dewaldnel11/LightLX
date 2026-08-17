from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict
    raw_arguments: str = ""


@dataclass
class Completion:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish: str = "stop"


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict
    handler: Callable[..., str]
    source: str = "builtin"

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def openai_tool(self) -> dict:
        return self.schema()


Message = dict[str, Any]
