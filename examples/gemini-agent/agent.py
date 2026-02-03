import os
from typing import Any, AsyncGenerator, Dict, List, Optional

import aiohttp
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types
from pydantic import ConfigDict
from vertexai.generative_models import Content, GenerativeModel, Part


class QontextAgent(BaseAgent):
    """Custom agent with query rewriting for contextual follow-up questions."""

    model_config = ConfigDict(extra='allow')

    model_name: str = "gemini-2.0-flash"
    project_id: Optional[str] = None
    
    api_key: Optional[str] = None
    vault_id: Optional[str] = None
    
    http_session: Any = None
    model_client: Any = None

    def __init__(self, **kwargs):
        """Initialize the Qontext agent with default configuration."""
        kwargs.setdefault("name", "qontext_agent_example")
        kwargs.setdefault("model_name", os.environ.get("MODEL_NAME", "gemini-2.0-flash"))
        kwargs.setdefault("project_id", os.environ.get("GCP_PROJECT_ID"))
        super().__init__(**kwargs)

    async def set_up(self) -> None:
        """Initialize the agent with required credentials and clients."""
        if not self.http_session:
            self.http_session = aiohttp.ClientSession()
        
        self.model_client = GenerativeModel(self.model_name)
        await self._load_from_secret_manager()

    async def _run_async_impl(
        self, 
        ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """
        Execute the agent's main logic with query rewriting.
        
        Steps:
        1. Initialize session state
        2. Build conversation history
        3. Rewrite query to resolve pronouns and context
        4. Query Qontext with the rewritten query
        5. Build enhanced prompt with context
        6. Generate and yield response
        """
        if self.api_key is None:
            await self.set_up()

        if self.model_client is None:
            self.model_client = GenerativeModel(self.model_name)
            
        if self.http_session is None or self.http_session.closed:
            self.http_session = aiohttp.ClientSession()

        if not hasattr(ctx.session, 'state') or ctx.session.state is None:
            ctx.session.state = {}
        
        user_message = self._extract_user_message(ctx)
        conversation_history = self._build_conversation_history(ctx)
        search_query = await self._rewrite_query(user_message, conversation_history)
        context_data = await self._query_qontext_tool(search_query)
        enhanced_message = self._build_enhanced_message(user_message, context_data)
        full_history = conversation_history + [
            Content(role="user", parts=[Part.from_text(enhanced_message)])
        ]
        
        response = await self.model_client.generate_content_async(
            contents=full_history,
            generation_config={"temperature": 0.7}
        )
        
        yield Event(
            author=self.name,
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text=response.text)]
            ),
            invocation_id=ctx.invocation_id
        )

    # Query Rewriting Methods
    async def _rewrite_query(
        self, 
        user_message: str, 
        history: List[Content]
    ) -> str:
        """
        Use LLM to resolve pronouns and context before searching.
        """
        if not history:
            return user_message

        history_str = self._format_history_for_rewrite(history)
        
        rewrite_prompt = f"""Given the following conversation history and a new user message, rewrite the user message to be a standalone search query. 
The goal is to replace pronouns (he, she, it, they, that) with the actual entities they refer to so the query can be used for a database search.

History:
{history_str}

User Message: {user_message}

Standalone Search Query (concise):"""

        try:
            response = await self.model_client.generate_content_async(
                rewrite_prompt,
                generation_config={"temperature": 0.0}
            )
            rewritten = response.text.strip()
            return rewritten
        except Exception:
            return user_message

    def _format_history_for_rewrite(self, history: List[Content]) -> str:
        """Format conversation history for the query rewriter."""
        formatted = []
        
        # Use last 4 turns for context
        for h in history[-4:]:
            role = "User" if h.role == "user" else "AI"
            text = h.parts[0].text
            
            # Truncate long responses
            if len(text) > 200:
                text = text[:200] + "..."
                
            formatted.append(f"{role}: {text}")
            
        return "\n".join(formatted)

    # Context Building Methods
    def _build_enhanced_message(
        self, 
        user_message: str, 
        context_data: Dict[str, Any]
    ) -> str:
        """Combine retrieval results with user prompt."""
        context_parts = []
        
        if context_data and "error" not in context_data:
            context_str = self._format_context_data(context_data)
            context_parts.append(f"Knowledge base information:\n{context_str}")
        
        if context_parts:
            context_section = "\n\n".join(context_parts)
            return f"Context from Knowledge Base:\n{context_section}\n\nUser Question: {user_message}"
        
        return user_message

    def _build_conversation_history(self, ctx: InvocationContext) -> List[Content]:
        """Reconstruct message history for the generator."""
        conversation_history = []
        
        if hasattr(ctx.session, 'events') and ctx.session.events:
            for event in ctx.session.events:
                if hasattr(event, 'content') and event.content:
                    if event.content.role in ("user", "model"):
                        parts = [
                            Part.from_text(p.text) 
                            for p in event.content.parts 
                            if hasattr(p, 'text') and p.text
                        ]
                        if parts:
                            conversation_history.append(
                                Content(role=event.content.role, parts=parts)
                            )
                            
        return conversation_history

    def _format_context_data(self, context_data: Dict[str, Any]) -> str:
        """Format Qontext results into a readable string."""
        # Handle direct list response
        if isinstance(context_data, list):
            return "\n".join([f"- {item}" for item in context_data])
        
        # Handle dict with results array
        if isinstance(context_data, dict) and "results" in context_data:
            results = context_data["results"]
            formatted = []
            for res in results:
                text = res if isinstance(res, str) else res.get("text", str(res))
                formatted.append(f"- {text}")
            return "\n".join(formatted)
        
        return str(context_data)

    # Qontext API Methods
    async def _query_qontext_tool(self, prompt: str) -> Dict[str, Any]:
        """Query Qontext API with the given prompt."""
        url = "https://api.qontext.ai/v1/retrieval"
        
        payload = {
            "knowledgeGraphId": str(self.vault_id),
            "prompt": str(prompt),
            "limit": 5,
            "depth": 1,
        }
        
        headers = {
            "X-API-Key": str(self.api_key),
            "Content-Type": "application/json"
        }
        
        try:
            async with self.http_session.post(
                url, 
                headers=headers, 
                json=payload, 
                timeout=15
            ) as response:
                if response.status == 201:
                    data = await response.json()
                    return self._sanitize_for_pydantic(data)
                
                error_text = await response.text()
                return {"error": error_text}
                
        except Exception as e:
            return {"error": str(e)}

    def _extract_user_message(self, ctx: InvocationContext) -> str:
        """Extract the user's message from the invocation context."""
        if ctx.user_content and ctx.user_content.parts:
            return ctx.user_content.parts[0].text
        return ""

    def _sanitize_for_pydantic(self, obj: Any) -> Any:
        """Recursively sanitize data for Pydantic compatibility."""
        if isinstance(obj, dict):
            return {str(k): self._sanitize_for_pydantic(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._sanitize_for_pydantic(i) for i in obj]
        return obj

    async def _load_from_secret_manager(self) -> None:
        """Load credentials from Google Cloud Secret Manager."""
        from google.cloud import secretmanager
        
        client = secretmanager.SecretManagerServiceClient()
        
        secret_map = {
            "api_key": "QONTEXT_API_KEY_SECRET",
            "vault_id": "QONTEXT_VAULT_ID_SECRET"
        }
        
        for attr, secret_id in secret_map.items():
            path = f"projects/{self.project_id}/secrets/{secret_id}/versions/latest"
            response = client.access_secret_version(request={"name": path})
            setattr(self, attr, response.payload.data.decode('UTF-8'))

root_agent = QontextAgent()