import asyncio
import aiohttp
import logging
from typing import Optional
from opentelemetry.trace import Span

from vocode.streaming.models.synthesizer import (
    DeepgramSynthesizerConfig,
    SynthesizerType,
)
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.agent.bot_sentiment_analyser import BotSentiment
from vocode.streaming.utils.mp3_helper import decode_mp3
from vocode.streaming.synthesizer.base_synthesizer import (
    BaseSynthesizer,
    SynthesisResult,
    encode_as_wav,
    tracer,
)
from vocode import getenv

DEEPGRAM_BASE_URL = "https://api.deepgram.com/v1/speak"


class DeepgramSynthesizer(BaseSynthesizer[DeepgramSynthesizerConfig]):
    def __init__(
        self,
        synthesizer_config: DeepgramSynthesizerConfig,
        logger: Optional[logging.Logger] = None,
        aiohttp_session: Optional[aiohttp.ClientSession] = None,
    ):
        super().__init__(synthesizer_config, aiohttp_session)
        self.api_key = synthesizer_config.api_key or getenv(
            "DEEPGRAM_SYNTHESIZER_API_KEY"
        )
        self.model = synthesizer_config.model

    async def create_speech(
        self,
        message: BaseMessage,
        chunk_size: int,
        bot_sentiment: Optional[BotSentiment] = None,
    ) -> SynthesisResult:
        url = f"{DEEPGRAM_BASE_URL}?model={self.model}"
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "text": message.text,
        }

        create_speech_span = tracer.start_span(
            f"synthesizer.{SynthesizerType.DEEPGRAM.value.split('_', 1)[-1]}.create_total",
        )

        session = self.aiohttp_session

        response = await session.request(
            "POST",
            url,
            json=body,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=15),
        )
        if not response.ok:
            raise Exception(f"Deepgram API returned {response.status} status code")

        audio_data = await response.read()
        create_speech_span.end()
        convert_span = tracer.start_span(
            f"synthesizer.{SynthesizerType.DEEPGRAM.value.split('_', 1)[-1]}.convert",
        )
        output_bytes_io = decode_mp3(audio_data)

        result = self.create_synthesis_result_from_wav(
            synthesizer_config=self.synthesizer_config,
            file=output_bytes_io,
            message=message,
            chunk_size=chunk_size,
        )
        convert_span.end()

        return result
