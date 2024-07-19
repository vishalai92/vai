from __future__ import annotations

import asyncio
import audioop
import base64
import json
from typing import List, Optional, Union

from fastapi import WebSocket
from fastapi.websockets import WebSocketState
from loguru import logger
from pydantic import BaseModel

from vocode.streaming.output_device.abstract_output_device import AbstractOutputDevice
from vocode.streaming.output_device.audio_chunk import AudioChunk, ChunkState
from vocode.streaming.telephony.constants import DEFAULT_AUDIO_ENCODING, DEFAULT_SAMPLING_RATE
from vocode.streaming.utils.create_task import asyncio_create_task
from vocode.streaming.utils.dtmf_utils import DTMFToneGenerator, KeypadEntry
from vocode.streaming.utils.worker import InterruptibleEvent


class ChunkFinishedMarkMessage(BaseModel):
    chunk_id: str


MarkMessage = Union[ChunkFinishedMarkMessage]  # space for more mark messages


class TwilioOutputDevice(AbstractOutputDevice):
    def __init__(self, ws: Optional[WebSocket] = None, stream_sid: Optional[str] = None):
        super().__init__(sampling_rate=DEFAULT_SAMPLING_RATE, audio_encoding=DEFAULT_AUDIO_ENCODING)
        self.ws = ws
        self.stream_sid = stream_sid
        self.active = True

        self._twilio_events_queue: asyncio.Queue[str] = asyncio.Queue()
        self._mark_message_queue: asyncio.Queue[MarkMessage] = asyncio.Queue()
        self._unprocessed_audio_chunks_queue: asyncio.Queue[InterruptibleEvent[AudioChunk]] = (
            asyncio.Queue()
        )

    def consume_nonblocking(self, item: InterruptibleEvent[AudioChunk]):
        if not item.is_interrupted():
            self._send_audio_chunk_and_mark(
                chunk=item.payload.data, chunk_id=str(item.payload.chunk_id)
            )
            self._unprocessed_audio_chunks_queue.put_nowait(item)
        else:
            audio_chunk = item.payload
            audio_chunk.on_interrupt()
            audio_chunk.state = ChunkState.INTERRUPTED

    def interrupt(self):
        self._send_clear_message()

    def enqueue_mark_message(self, mark_message: MarkMessage):
        self._mark_message_queue.put_nowait(mark_message)

    def send_dtmf_tones(self, keypad_entries: List[KeypadEntry]):
        tone_generator = DTMFToneGenerator()
        for keypad_entry in keypad_entries:
            logger.info(f"Sending DTMF tone {keypad_entry.value}")
            dtmf_tone = tone_generator.generate(
                keypad_entry, sampling_rate=self.sampling_rate, audio_encoding=self.audio_encoding
            )
            dtmf_message = {
                "event": "media",
                "streamSid": self.stream_sid,
                "media": {"payload": base64.b64encode(dtmf_tone).decode("utf-8")},
            }
            self._twilio_events_queue.put_nowait(json.dumps(dtmf_message))

    async def _send_twilio_messages(self):
        while True:
            try:
                twilio_event = await self._twilio_events_queue.get()
            except asyncio.CancelledError:
                return
            if self.ws.application_state == WebSocketState.DISCONNECTED:
                break
            await self.ws.send_text(twilio_event)

    async def _process_mark_messages(self):
        while True:
            try:
                # mark messages are tagged with the chunk ID that is attached to the audio chunk
                # but they are guaranteed to come in the same order as the audio chunks, and we
                # don't need to build resiliency there
                mark_message = await self._mark_message_queue.get()
                item = await self._unprocessed_audio_chunks_queue.get()
            except asyncio.CancelledError:
                return

            self.interruptible_event = item
            audio_chunk = item.payload

            if mark_message.chunk_id != str(audio_chunk.chunk_id):
                logger.error(
                    f"Received a mark message out of order with chunk ID {mark_message.chunk_id}"
                )

            if item.is_interrupted():
                audio_chunk.on_interrupt()
                audio_chunk.state = ChunkState.INTERRUPTED
                continue

            audio_chunk.on_play()
            audio_chunk.state = ChunkState.PLAYED

            self.interruptible_event.is_interruptible = False

    async def _run_loop(self):
        send_twilio_messages_task = asyncio_create_task(self._send_twilio_messages())
        process_mark_messages_task = asyncio_create_task(self._process_mark_messages())
        await asyncio.gather(send_twilio_messages_task, process_mark_messages_task)

    def _send_audio_chunk_and_mark(self, chunk: bytes, chunk_id: str):
        media_message = {
            "event": "media",
            "streamSid": self.stream_sid,
            "media": {"payload": base64.b64encode(chunk).decode("utf-8")},
        }
        self._twilio_events_queue.put_nowait(json.dumps(media_message))
        mark_message = {
            "event": "mark",
            "streamSid": self.stream_sid,
            "mark": {
                "name": chunk_id,
            },
        }
        self._twilio_events_queue.put_nowait(json.dumps(mark_message))

    def _send_clear_message(self):
        clear_message = {
            "event": "clear",
            "streamSid": self.stream_sid,
        }
        self._twilio_events_queue.put_nowait(json.dumps(clear_message))
