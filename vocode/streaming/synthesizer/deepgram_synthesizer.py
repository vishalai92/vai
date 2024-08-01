import asyncio
import aiohttp
import logging
from typing import Optional, Tuple, Dict
from opentelemetry.trace import Span

from vocode.streaming.models.synthesizer import (
    DeepgramSynthesizerConfig,
    SynthesizerType,
)
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.utils.mp3_helper import decode_mp3
from vocode.streaming.synthesizer.base_synthesizer import (
    BaseSynthesizer,
    SynthesisResult,
    encode_as_wav,
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

    def get_request(self, text: str) -> Tuple[str, Dict[str, str], Dict[str, object]]:
        url = f"{DEEPGRAM_BASE_URL}?model={self.model}"
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "text": text,
        }
        return url, headers, body

    async def create_speech(
        self,
        message: BaseMessage,
        chunk_size: int,
        is_first_text_chunk: bool = False,
        is_sole_text_chunk: bool = False,
    ) -> SynthesisResult:
        url, headers, body = self.get_request(message.text)

        async with self.async_requestor.get_session().request(
            "POST",
            url,
            json=body,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as response:
            if not response.ok:
                raise Exception(f"Deepgram API returned {response.status} status code")

            audio_data = await response.read()
            output_bytes_io = decode_mp3(audio_data)

            result = self.create_synthesis_result_from_wav(
                synthesizer_config=self.synthesizer_config,
                file=output_bytes_io,
                message=message,
                chunk_size=chunk_size,
            )

            return result
