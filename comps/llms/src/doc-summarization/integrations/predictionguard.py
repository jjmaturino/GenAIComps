# Copyright (C) 2025 Prediction Guard, Inc.
# SPDX-License-Identified: Apache-2.0

import os
import time

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from predictionguard import PredictionGuard

from comps import CustomLogger, OpeaComponent, OpeaComponentRegistry, ServiceType
from comps.cores.proto.api_protocol import DocSumChatCompletionRequest

logger = CustomLogger("opea_docsum_predictionguard")
logflag = os.getenv("LOGFLAG", False)


@OpeaComponentRegistry.register("OPEA_PREDICTIONGUARD_DOCSUM")
class PredictionguardDocsum(OpeaComponent):
    """A specialized docsum component derived from OpeaComponent for interacting with Prediction Guard services.

    Attributes:
        client (PredictionGuard): An instance of the PredictionGuard client for document summarization.
        model_name (str): The name of the llm model used by the Prediction Guard service.
    """

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.LLM.name.lower(), description, config)
        api_key = os.getenv("PREDICTIONGUARD_API_KEY")
        self.client = None
        if api_key:
            self.client = PredictionGuard(api_key=api_key)
        else:
            logger.info("No PredictionGuard API KEY provided, client not instantiated")

        health_status = self.check_health()
        if not health_status:
            logger.error("OpeaDocSumPredictionguard health check failed.")

    def check_health(self) -> bool:
        """Checks the health of the Predictionguard LLM service.

        Returns:
            bool: True if the service is reachable and healthy, False otherwise.
        """

        try:
            if not self.client:
                return False

            response = self.client.models.list()
            return response is not None
        except Exception as e:

            # Handle exceptions such as network errors or unexpected failures
            logger.error(e)
            logger.error(f"Health check failed due to an exception: {e}")

            return False

    async def invoke(self, input: DocSumChatCompletionRequest):
        """Invokes the Predictionguard LLM service to generate output for the provided input.

        Args:
            input (ChatCompletionRequest): The input text(s).
        """

        res =  await self.generate(input)

        return res

    async def generate(self, input: DocSumChatCompletionRequest):
        """ Calls the Predictionguard LLM service to generate output for the provided input. """
        if isinstance(input.messages, str):
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Your goal is to provide accurate, detailed, and safe responses to the user's queries.",
                },
                {"role": "user", "content": input.messages},
            ]
        else:
            messages = input.messages

        if input.stream:

            async def stream_generator():
                chat_response = ""
                for res in self.client.chat.completions.create(
                        model=input.model,
                        messages=messages,
                        max_tokens=input.max_tokens,
                        temperature=input.temperature,
                        top_p=input.top_p,
                        top_k=input.top_k,
                        stream=True,
                ):
                    if "choices" in res["data"] and "delta" in res["data"]["choices"][0]:
                        delta_content = res["data"]["choices"][0]["delta"]["content"]
                        chat_response += delta_content
                        yield f"data: {delta_content}\n\n"
                    else:
                        yield "data: [DONE]\n\n"

            return StreamingResponse(stream_generator(), media_type="text/event-stream")
        else:
            try:
                response = self.client.chat.completions.create(
                    model=input.model,
                    messages=messages,
                    max_tokens=input.max_tokens,
                    temperature=input.temperature,
                    top_p=input.top_p,
                    top_k=input.top_k,
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

            return response
