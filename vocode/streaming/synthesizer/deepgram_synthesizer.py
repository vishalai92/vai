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

        async with session.post(
            url, json=body, headers=headers, timeout=aiohttp.ClientTimeout(total=15)
        ) as response:
            if response.status != 200:
                raise Exception(f"Deepgram API returned {response.status} status code")

            return SynthesisResult(
                self.experimental_mp3_streaming_output_generator(
                    response, chunk_size, create_speech_span
                ),  # should be wav
                lambda seconds: self.get_message_cutoff_from_voice_speed(
                    message, seconds, self.words_per_minute
                ),
            )


# Additional classes like DeepgramSynthesizerConfig, BotSentiment might need to be defined or adapted based on the existing code structure.
