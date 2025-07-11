# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import base64
import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Any, List, Literal

import aiohttp
from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
    APIConnectOptions,
)

from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)

from .log import logger
from .models import TTSEncoding, TTSModels
# from dotenv import load_dotenv
# load_dotenv()

_Encoding = Literal["mp3", "pcm"]


word_tokenizer_without_punctuation: tokenize.WordTokenizer = (
    tokenize.basic.WordTokenizer(ignore_punctuation=True)
)


trigger_phrase = os.environ.get("XTTS_TRIGGER_PHRASE", "Hi Agent")
trigger_phrase_words = word_tokenizer_without_punctuation.tokenize(text=trigger_phrase)


def _sample_rate_from_format(output_format: TTSEncoding) -> int:
    split = output_format.split("_")  # e.g: mp3_22050_32
    return int(split[1])


def _encoding_from_format(output_format: TTSEncoding) -> _Encoding:
    if output_format.startswith("mp3"):
        return "mp3"
    elif output_format.startswith("pcm"):
        return "pcm"

    raise ValueError(f"Unknown format: {output_format}")


@dataclass
class VoiceSettings:
    stability: float  # [0.0 - 1.0]
    similarity_boost: float  # [0.0 - 1.0]
    style: float | None = None  # [0.0 - 1.0]
    use_speaker_boost: bool | None = False


@dataclass
class Voice:
    id: str
    name: str
    category: str
    settings: VoiceSettings | None = None




API_BASE_URL_V1 = os.environ.get("XTTS_BASE_URL", "http://localhost:8020")
AUTHORIZATION_HEADER = "xi-api-key"


@dataclass
class _TTSOptions:
    api_key: str | None
    voice: Voice | None
    model: TTSModels | str
    language: str | None
    base_url: str
    encoding: TTSEncoding
    sample_rate: int 
    streaming_latency: int 
    word_tokenizer: tokenize.WordTokenizer
    chunk_length_schedule: List[int] = field(default_factory=list)
    enable_ssml_parsing: bool = False


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: Voice,
        model: TTSModels | str = "sample_model",
        api_key: str | None = None,
        base_url: str | None = None,
        encoding: TTSEncoding = "mp3_22050_32",
        streaming_latency: int = 3,
        word_tokenizer: tokenize.WordTokenizer = trigger_phrase_words,
        enable_ssml_parsing: bool = False,
        chunk_length_schedule: List[int] = [80, 120, 200, 260],  # range is [50, 500]
        http_session: aiohttp.ClientSession | None = None,
        # deprecated
        model_id: TTSModels | str | None = None,
        language: str | None = None,
    ) -> None:
        """
        Create a new instance of XTTS.

        Args:
            model (TTSModels | str): TTS model to use. Defaults to "male.wav".
            base_url (str | None): Custom base URL for the API. Optional.
            encoding (TTSEncoding): Audio encoding format. Defaults to "mp3_22050_32".
            streaming_latency (int): Latency in seconds for streaming. Defaults to 3.
            word_tokenizer (tokenize.WordTokenizer): Tokenizer for processing text. Defaults to basic WordTokenizer.
            enable_ssml_parsing (bool): Enable SSML parsing for input text. Defaults to False.
            chunk_length_schedule (list[int]): Schedule for chunk lengths, ranging from 50 to 500. Defaults to [80, 120, 200, 260].
            http_session (aiohttp.ClientSession | None): Custom HTTP session for API requests. Optional.
            language (str | None): Language code for the TTS model, as of 10/24/24 only valid for "eleven_turbo_v2_5". Optional.
        """

        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,
            ),
            sample_rate=_sample_rate_from_format(encoding),
            num_channels=1,
        )

        if model_id is not None:
            logger.warning(
                "model_id is deprecated and will be removed in 1.5.0, use model instead",
            )
            model = model_id

        api_key = api_key or os.environ.get("XTTS_API_KEY")
        if not api_key:
            api_key = "1234567890"

        self._opts = _TTSOptions(
            voice=voice,
            model=model,
            api_key=api_key,
            base_url=base_url or API_BASE_URL_V1,
            encoding=encoding,
            sample_rate=self.sample_rate,
            streaming_latency=streaming_latency,
            word_tokenizer=word_tokenizer,
            chunk_length_schedule=chunk_length_schedule,
            enable_ssml_parsing=enable_ssml_parsing,
            language=language,
        )
        self._session = http_session

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    async def list_voices(self) -> List[Voice]:
        async with self._ensure_session().get(
            f"{self._opts.base_url}/voices",
            headers={AUTHORIZATION_HEADER: self._opts.api_key},
        ) as resp:
            return _dict_to_voices_list(await resp.json())

    def update_options(
        self,
        *,
        model: TTSModels | str = "sample_model",
    ) -> None:
        """
        Args:
            voice (Voice): Voice configuration. Defaults to `DEFAULT_VOICE`.
            model (TTSModels | str): TTS model to use. Defaults to "eleven_turbo_v2_5".
        """
        self._opts.model = model or self._opts.model
        

    def synthesize(self, text: str, conn_options) -> "ChunkedStream":
        return ChunkedStream(self, text, self._opts, self._ensure_session())

    # def stream(self) -> "SynthesizeStream":
    #     return SynthesizeStream(self, self._ensure_session(), self._opts)


class ChunkedStream(tts.ChunkedStream):
    """Synthesize using the chunked api endpoint"""

    def __init__(
        self, tts: TTS, input_text: str, opts: _TTSOptions, session: aiohttp.ClientSession
    ) -> None:
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts, self._session = opts, session
        if _encoding_from_format(self._opts.encoding) == "mp3":
            self._mp3_decoder = utils.codecs.Mp3StreamDecoder()

    async def _run(self, output_emitter) -> None:
        # Skip TTS if input text is empty, whitespace, or any form of empty string representation
        if (not self._input_text or 
            self._input_text.isspace() or 
            self._input_text.strip() in ['""', "''", ''] or
            self._input_text.strip() == '""'):
            return

        request_id = utils.shortuuid()
        bstream = utils.audio.AudioByteStream(
            sample_rate=self._opts.sample_rate, num_channels=1
        )

        voice_settings = (
            _strip_nones(dataclasses.asdict(self._opts.voice.settings))
            if self._opts.voice.settings
            else None
        )
        data = {
            "text": self._input_text,
            "model_id": self._opts.model,
            "voice_settings": voice_settings,
        }

        try:
            async with self._session.get(
                _synthesize_url(self._opts, self._input_text),
                headers={AUTHORIZATION_HEADER: self._opts.api_key},
            ) as resp:
                # if not resp.content_type.startswith("audio/"):
                #     content = await resp.text()
                #     logger.error("11labs returned non-audio data: %s", content)
                #     return

                # encoding = _encoding_from_format(self._opts.encoding)
                # if encoding == "mp3":
                #     async for bytes_data, _ in resp.content.iter_chunks():
                #         for frame in self._mp3_decoder.decode_chunk(bytes_data):
                #             for frame in bstream.write(frame.data.tobytes()):
                #                 self._event_ch.send_nowait(
                #                     tts.SynthesizedAudio(
                #                         request_id=request_id,
                #                         frame=frame,
                #                     )
                #                 )
                # else:
                print("resp.content", resp.content)
                async for bytes_data, _ in resp.content.iter_chunks():
                    for frame in bstream.write(bytes_data):
                        self._event_ch.send_nowait(
                            tts.SynthesizedAudio(
                                request_id=request_id,
                                segment_id=request_id,
                                frame=frame,
                            )
                        )

                for frame in bstream.flush():
                    self._event_ch.send_nowait(
                        tts.SynthesizedAudio(request_id=request_id, frame=frame)
                    )

        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=None,
                body=None,
            ) from e
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    """Streamed API using websockets"""

    def __init__(
        self,
        tts: TTS,
        session: aiohttp.ClientSession,
        opts: _TTSOptions,
    ):
        super().__init__(tts)
        self._opts, self._session = opts, session
        self._mp3_decoder = utils.codecs.Mp3StreamDecoder()

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        self._segments_ch = utils.aio.Chan[tokenize.WordStream]()

        @utils.log_exceptions(logger=logger)
        async def _tokenize_input():
            """tokenize text from the input_ch to words"""
            word_stream = None
            async for input in self._input_ch:
                if isinstance(input, str):
                    if word_stream is None:
                        # new segment (after flush for e.g)
                        word_stream = self._opts.word_tokenizer.stream()
                        self._segments_ch.send_nowait(word_stream)

                    word_stream.push_text(input)
                elif isinstance(input, self._FlushSentinel):
                    if word_stream is not None:
                        word_stream.end_input()

                    word_stream = None

            self._segments_ch.close()

        @utils.log_exceptions(logger=logger)
        async def _run():
            async for word_stream in self._segments_ch:
                await self._run_ws(word_stream)

        tasks = [
            asyncio.create_task(_tokenize_input()),
            asyncio.create_task(_run()),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.gracefully_cancel(*tasks)

    async def _run_ws(
        self,
        word_stream: tokenize.WordStream,
        max_retry: int = 3,
    ) -> None:
        ws_conn: aiohttp.ClientWebSocketResponse | None = None
        for try_i in range(max_retry):
            retry_delay = 5
            try:
                if try_i > 0:
                    await asyncio.sleep(retry_delay)

                ws_conn = await self._session.ws_connect(
                    _stream_url(self._opts),
                    headers={AUTHORIZATION_HEADER: self._opts.api_key},
                )
                break
            except Exception as e:
                logger.warning(
                    f"failed to connect to 11labs, retrying in {retry_delay}s",
                    exc_info=e,
                )

        if ws_conn is None:
            raise Exception(f"failed to connect to 11labs after {max_retry} retries")

        request_id = utils.shortuuid()
        segment_id = utils.shortuuid()

        # 11labs protocol expects the first message to be an "init msg"
        init_pkt = dict(
            text=" ",
            try_trigger_generation=True,
            voice_settings=_strip_nones(dataclasses.asdict(self._opts.voice.settings))
            if self._opts.voice.settings
            else None,
            generation_config=dict(
                chunk_length_schedule=self._opts.chunk_length_schedule
            ),
        )
        await ws_conn.send_str(json.dumps(init_pkt))
        eos_sent = False

        async def send_task():
            nonlocal eos_sent

            xml_content = []
            async for data in word_stream:
                text = data.token

                # send the xml phoneme in one go
                if (
                    self._opts.enable_ssml_parsing
                    and data.token.startswith("<phoneme")
                    or xml_content
                ):
                    xml_content.append(text)
                    if data.token.find("</phoneme>") > -1:
                        text = self._opts.word_tokenizer.format_words(xml_content)
                        xml_content = []
                    else:
                        continue

                # try_trigger_generation=True is a bad practice, we expose
                # chunk_length_schedule instead
                data_pkt = dict(
                    text=f"{text} ",  # must always end with a space
                    try_trigger_generation=False,
                )
                await ws_conn.send_str(json.dumps(data_pkt))

            if xml_content:
                logger.warning("11labs stream ended with incomplete xml content")

            # no more token, mark eos
            eos_pkt = dict(text="")
            await ws_conn.send_str(json.dumps(eos_pkt))
            eos_sent = True

        async def recv_task():
            nonlocal eos_sent
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=1,
            )

            last_frame: rtc.AudioFrame | None = None

            def _send_last_frame(*, segment_id: str, is_final: bool) -> None:
                nonlocal last_frame
                if last_frame is not None:
                    self._event_ch.send_nowait(
                        tts.SynthesizedAudio(
                            request_id=request_id,
                            segment_id=segment_id,
                            frame=last_frame,
                            is_final=is_final,
                        )
                    )

                    last_frame = None

            while True:
                msg = await ws_conn.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if not eos_sent:
                        raise Exception(
                            "11labs connection closed unexpectedly, not all tokens have been consumed"
                        )
                    return

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected 11labs message type %s", msg.type)
                    continue

                data = json.loads(msg.data)
                encoding = _encoding_from_format(self._opts.encoding)
                if data.get("audio"):
                    b64data = base64.b64decode(data["audio"])
                    if encoding == "mp3":
                        for frame in self._mp3_decoder.decode_chunk(b64data):
                            for frame in audio_bstream.write(frame.data.tobytes()):
                                _send_last_frame(segment_id=segment_id, is_final=False)
                                last_frame = frame

                    else:
                        for frame in audio_bstream.write(b64data):
                            _send_last_frame(segment_id=segment_id, is_final=False)
                            last_frame = frame

                elif data.get("isFinal"):
                    for frame in audio_bstream.flush():
                        _send_last_frame(segment_id=segment_id, is_final=False)
                        last_frame = frame

                    _send_last_frame(segment_id=segment_id, is_final=True)

                    pass
                elif data.get("error"):
                    logger.error("11labs reported an error: %s", data["error"])
                else:
                    logger.error("unexpected 11labs message %s", data)

        tasks = [
            asyncio.create_task(send_task()),
            asyncio.create_task(recv_task()),
        ]

        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.gracefully_cancel(*tasks)


def _dict_to_voices_list(data: dict[str, Any]):
    voices: List[Voice] = []
    for voice in data["voices"]:
        voices.append(
            Voice(
                id=voice["voice_id"],
                name=voice["name"],
                category=voice["category"],
                settings=None,
            )
        )
    return voices


def _strip_nones(data: dict[str, Any]):
    return {k: v for k, v in data.items() if v is not None}


def _synthesize_url(opts: _TTSOptions,text:str) -> str:
    import urllib.parse
    default_speaker = os.environ.get("XTTS_SPEAKER", "male.wav")
    default_language = os.environ.get("XTTS_LANGUAGE", "en")
    safe_string = urllib.parse.quote(text)
    base_url = f"{opts.base_url}/tts_stream?text={safe_string}&speaker_wav={default_speaker}&language={default_language}"
    return (
        f"{base_url}"
    )


def _stream_url(opts: _TTSOptions) -> str:
    base_url = opts.base_url
    voice_id = opts.voice.id
    model_id = opts.model
    output_format = opts.encoding
    latency = opts.streaming_latency
    enable_ssml = str(opts.enable_ssml_parsing).lower()
    language = opts.language
    url = (
        f"{base_url}/text-to-speech/{voice_id}/stream-input?"
        f"model_id={model_id}&output_format={output_format}&optimize_streaming_latency={latency}&"
        f"enable_ssml_parsing={enable_ssml}"
    )
    if language is not None:
        url += f"&language_code={language}"
    return url
