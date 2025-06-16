from typing import Type

from crewai.tools import BaseTool, tool
from pydantic import BaseModel, Field

from utils.documentation import DocumentationService


class DocQueryInput(BaseModel):
    """Input schema asking for documentation of a MAVLink (ArduPilot) message type."""
    message_type: str = Field(..., description="ArduPilot log message type, e.g. 'GPS', 'ATT', 'CURR'")


class MessageDocumentationTool(BaseTool):
    """Return human-readable documentation for a given ArduPilot log message type."""

    name: str = "get_message_doc"
    description: str = (
        "Fetch concise documentation for any ArduPilot/ MAVLink log message type so the LLM can explain column meanings to the user. "
        "Provide a single message_type such as 'GPS', 'BAT', 'CTUN'."
    )
    args_schema: Type[BaseModel] = DocQueryInput

    def __init__(self, doc_service: DocumentationService):
        super().__init__()
        self._doc_service = doc_service

    def _run(self, message_type: str) -> str:
        """Return cached or static documentation for the requested message type."""
        try:
            # Try cached dynamic documentation; fall back to static dict
            msg = message_type.strip().upper()
            # First attempt asynchronous fetch in a blocking way only if not already cached
            if msg in self._doc_service.ardupilot_messages:
                return self._doc_service.ardupilot_messages[msg]

            # If not in static dict, attempt network fetch (non-blocking fallback)
            import asyncio
            try:
                doc = asyncio.run(self._doc_service.fetch_ardupilot_documentation(msg))
            except RuntimeError:
                # Already in an event loop; run synchronously
                doc = asyncio.get_event_loop().run_until_complete(
                    self._doc_service.fetch_ardupilot_documentation(msg)
                )
            if doc:
                return doc
        except Exception:
            pass
        return f"No documentation available for message type '{message_type.upper()}'." 