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

# Imports the 'annotations' feature from the future, allowing the use of type hints in a more flexible way.
from __future__ import annotations

# Imports the 'asyncio' library for writing asynchronous code using coroutines, which is essential for handling concurrent operations like network requests.
import asyncio

# Imports the 'base64' library for encoding and decoding binary data into ASCII strings and vice versa, often used for transmitting binary data over text-based protocols.
import base64

# Imports the 'dataclasses' module for easily creating classes that primarily store data, with automatic generation of methods like '__init__', '__repr__', etc.
import dataclasses

# Imports the 'json' library for working with JSON (JavaScript Object Notation) data, a common format for data exchange on the web.
import json

# Imports the 'os' module for interacting with the operating system, such as accessing environment variables or file system operations.
import os

# Imports 'dataclass' and 'field' from the 'dataclasses' module to define data classes with default values or other configurations for their fields.
from dataclasses import dataclass, field

# Imports 'Any', 'List', and 'Literal' from the 'typing' module for type hinting, providing more information about the expected types of variables and function arguments.
from typing import Any, List, Literal, Union 

# Imports the 'aiohttp' library for making asynchronous HTTP requests, which is crucial for interacting with web APIs.
import aiohttp

# Imports the 'rtc' module from the 'livekit' library, likely related to real-time communication functionalities.
from livekit import rtc

# Imports several modules from the 'livekit.agents' library, including functionalities for handling API connections, tokenization, text-to-speech (TTS), utilities, and API connection options.
from livekit.agents import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
    APIConnectOptions,
    APIError,
)

# Imports specific types and default configurations for API connections from the 'livekit.agents.types' module.
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)

from livekit.agents.utils import aio, is_given  # Imports utility functions from livekit-agents
# Imports the 'logger' object from the '.log' module (likely a custom logging setup) for logging messages within this module.
from .log import logger

# Imports 'TTSEncoding' and 'TTSModels' from the '.models' module, which probably define the supported TTS encoding formats and model types as type hints.
from .models import TTSEncoding, TTSModels

# These lines are commented out.  They would have imported the 'load_dotenv' function (likely from the 'python-dotenv' package) for loading environment variables from a '.env' file.
# from dotenv import load_dotenv
# load_dotenv()


# Defines a type alias '_Encoding' using Literal to restrict possible values to "mp3" or "pcm", representing audio encoding formats.
_Encoding = Literal["mp3", "pcm"]

# Creates a WordTokenizer instance that ignores punctuation during tokenization.
word_tokenizer_without_punctuation: tokenize.WordTokenizer = (
    tokenize.basic.WordTokenizer(ignore_punctuation=True)
)

# Retrieves the value of the environment variable 'XTTS_TRIGGER_PHRASE', defaulting to "Hi Agent" if not set.
trigger_phrase = os.environ.get("XTTS_TRIGGER_PHRASE", "Hi Agent")

# Tokenizes the trigger phrase (without punctuation) using the previously defined tokenizer and stores the resulting tokens.
trigger_phrase_words = word_tokenizer_without_punctuation.tokenize(text=trigger_phrase)



def _sample_rate_from_format(output_format: TTSEncoding) -> int:
    # Splits the output_format string (e.g., "mp3_22050_32") into a list of substrings using "_" as the delimiter.
    split = output_format.split("_")  # e.g: mp3_22050_32
    # Returns the second element of the split list (index 1), which represents the sample rate, converted to an integer.
    return int(split[1])


def _encoding_from_format(output_format: TTSEncoding) -> _Encoding:
    # Checks if the output_format string starts with "mp3".
    if output_format.startswith("mp3"):
        # If it starts with "mp3", returns the string "mp3" indicating MP3 encoding.
        return "mp3"
    # Otherwise, checks if the output_format string starts with "pcm".
    elif output_format.startswith("pcm"):
        # If it starts with "pcm", returns the string "pcm" indicating PCM encoding.
        return "pcm"

    # If the output_format doesn't start with either "mp3" or "pcm", raises a ValueError with a descriptive message.
    raise ValueError(f"Unknown format: {output_format}")


@dataclass
class VoiceSettings:
    # A value between 0.0 and 1.0 representing the stability of the generated voice.
    stability: float  # [0.0 - 1.0]
    # A value between 0.0 and 1.0 representing the similarity boost for the generated voice.
    similarity_boost: float  # [0.0 - 1.0]
    # An optional value between 0.0 and 1.0 representing the style of the generated voice.
    style: float | None = None  # [0.0 - 1.0]
    # An optional boolean indicating whether to use speaker boost. Defaults to False if not provided.
    use_speaker_boost: bool | None = False


@dataclass
class Voice:
    # Unique identifier for the voice.
    id: str
    # Human-readable name of the voice.
    name: str
    # Category the voice belongs to (e.g., "generated", "cloned").
    category: str
    # Optional settings for the voice, allowing customization of voice characteristics.
    settings: VoiceSettings | None = None

# Retrieves the base URL for the XTTS API from the environment variable 'XTTS_BASE_URL'.
# If the environment variable is not set, it defaults to "http://localhost:8020".
API_BASE_URL_V1 = os.environ.get("XTTS_BASE_URL", "http://localhost:8020")
# Defines the header name used for API authorization, which is "xi-api-key".
AUTHORIZATION_HEADER = "xi-api-key"

@dataclass
class _TTSOptions:
    # API key for authentication (optional).
    api_key: str | None
    # Voice to be used for TTS (optional).
    voice: Voice | None
    # TTS model to use (can be a specific model or a category/string).
    model: TTSModels | str
    # Language code for the TTS model (optional).
    language: str | None
    # Base URL for the TTS API.
    base_url: str
    # Audio encoding format.
    encoding: TTSEncoding
    # Sample rate of the audio.
    sample_rate: int
    # Latency for streaming in seconds.
    streaming_latency: int
    # Tokenizer for processing text.
    word_tokenizer: tokenize.WordTokenizer
    # Schedule for chunk lengths during processing, defaults to an empty list.
    chunk_length_schedule: List[int] = field(default_factory=list)
    # Enable SSML parsing for input text, defaults to False.
    enable_ssml_parsing: bool = False

class TTS(tts.TTS):
    # Defines a class `TTS` that inherits from `tts.TTS`.  This likely represents a Text-to-Speech service.
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
        # Initializes a new instance of the `TTS` class.  It takes various parameters to configure the TTS service.
        #   `voice`: Specifies the voice to use for synthesis.
        #   `model`: The TTS model to use, defaults to "sample_model". Can also be a string.
        #   `api_key`: API key for authentication, optional.
        #   `base_url`: Custom base URL for the API, optional.
        #   `encoding`: Audio encoding format, defaults to "mp3_22050_32".
        #   `streaming_latency`: Latency in seconds for streaming, defaults to 3.
        #   `word_tokenizer`: Tokenizer for processing text, defaults to `trigger_phrase_words`.
        #   `enable_ssml_parsing`: Enables SSML parsing for input text, defaults to False.
        #   `chunk_length_schedule`:  Specifies the schedule for chunk lengths, with a default list and range.
        #   `http_session`: Custom HTTP session for API requests, optional.
        #   `model_id`: Deprecated alias for `model`.
        #   `language`: Language code for the TTS model, optional.

        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,
            ),
            sample_rate=_sample_rate_from_format(encoding),
            num_channels=1,
        )
        # Calls the constructor of the parent class (`tts.TTS`) to initialize common TTS properties.
        #   `capabilities`: Sets the TTS capabilities, in this case, streaming is disabled.
        #   `sample_rate`:  Determines the sample rate from the provided `encoding` using a helper function.
        #   `num_channels`: Sets the number of audio channels to 1 (mono).

        if model_id is not None:
            logger.warning(
                "model_id is deprecated and will be removed in 1.5.0, use model instead",
            )
            model = model_id
        # Handles the deprecated `model_id` parameter. If provided, it logs a warning and assigns its value to the `model` variable.

        api_key = api_key or os.environ.get("XTTS_API_KEY")
        if not api_key:
            api_key = "1234567890"
        # Determines the API key. It uses the provided `api_key` if given, otherwise tries to get it from the environment variable "XTTS_API_KEY". If neither is available, it defaults to "1234567890".

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
        # Creates an instance of the internal `_TTSOptions` dataclass to store the TTS configuration parameters.
        #  It uses the provided parameters or their defaults, including the base URL which defaults to `API_BASE_URL_V1` if not provided.
        #  It uses the sample_rate calculated in the super class init

        self._session = http_session
        # Stores the provided `http_session` for making API requests.


    def _ensure_session(self) -> aiohttp.ClientSession:
        # Checks if an HTTP client session already exists.
        if not self._session:
            # If a session doesn't exist, it creates a new one using the utils.http_context.http_session() function.
            self._session = utils.http_context.http_session()

        # Returns the existing or newly created HTTP client session.
        return self._session


    async def list_voices(self) -> List[Voice]:
        # Makes an asynchronous GET request to the /voices endpoint of the API.
        async with self._ensure_session().get(
            f"{self._opts.base_url}/voices",
            headers={AUTHORIZATION_HEADER: self._opts.api_key},
        ) as resp:
            # Parses the JSON response from the API and converts it into a list of Voice objects.
            return _dict_to_voices_list(await resp.json())


    def update_options(
        self,
        *,
        model: TTSModels | str = "sample_model",
    ) -> None:
        # This method updates the TTS options.
        """
        Args:
            voice (Voice): Voice configuration. Defaults to `DEFAULT_VOICE`.
            model (TTSModels | str): TTS model to use. Defaults to "eleven_turbo_v2_5".
        """
        #  Updates the model in the internal options (_opts).
        #  If a new model is provided, it will be used; otherwise, the existing model in _opts will be kept.
        self._opts.model = model or self._opts.model

        

    def synthesize(self, text: str, conn_options) -> "ChunkedStream":
        # Creates and returns a ChunkedStream object for synthesizing text using a chunked API endpoint.
        # It takes the TTS object itself (self), the text to synthesize, the TTS options, and an HTTP session as arguments.
        return ChunkedStream(self, text, self._opts, self._ensure_session())


    # def stream(self) -> "SynthesizeStream":
    #     return SynthesizeStream(self, self._ensure_session(), self._opts)


class ChunkedStream(tts.ChunkedStream):
    """Synthesize using the chunked api endpoint"""

    def __init__(
        self, tts: TTS, input_text: str, opts: _TTSOptions, session: aiohttp.ClientSession
    ) -> None:
        # Initializes the ChunkedStream object.
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
        # Sets the connection options to the default values.
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        # Calls the constructor of the parent class (tts.ChunkedStream) with the provided arguments.
        self._opts, self._session = opts, session
        # Stores the TTS options and the aiohttp client session as private attributes.
        # if _encoding_from_format(self._opts.encoding) == "mp3":
        #     self._mp3_decoder = utils.codecs.Mp3StreamDecoder()
        # If the specified encoding is MP3, initializes an MP3 stream decoder.

    async def _run(self, output_emitter) -> None:
        # Defines an asynchronous method to handle the synthesis process.
        # Skip TTS if input text is empty, whitespace, or any form of empty string representation
        if (not self._input_text or 
            self._input_text.isspace() or 
            self._input_text.strip() in ['""', "''", ''] or
            self._input_text.strip() == '""'):
            return
        # Checks if the input text is empty or contains only whitespace and returns immediately if so, skipping the TTS process.

        request_id = utils.shortuuid()
        # Generates a unique ID for the request.
        bstream = utils.audio.AudioByteStream(
            sample_rate=self._opts.sample_rate, num_channels=1
        )
        # Creates an audio byte stream with the specified sample rate and number of channels (mono).

        voice_settings = (
            _strip_nones(dataclasses.asdict(self._opts.voice.settings))
            if self._opts.voice.settings
            else None
        )
        # Extracts voice settings (stability, similarity_boost, style, use_speaker_boost) from the options, handling None values.
        data = {
            "text": self._input_text,
            "model_id": self._opts.model,
            "voice_settings": voice_settings,
        }
        # Constructs a dictionary containing the input text, model ID, and voice settings to be sent to the API.

        try:
            print("the url is: ",_synthesize_url(self._opts, self._input_text))
            async with self._session.get(
                _synthesize_url(self._opts, self._input_text),
                headers={AUTHORIZATION_HEADER: self._opts.api_key},
            ) as resp:
                # Sends an asynchronous GET request to the TTS API endpoint with the constructed data and authorization header.
                # if not resp.content_type.startswith("audio/"):
                #     content = await resp.text()
                #     logger.error("11labs returned non-audio data: %s", content)
                #     return
                if not resp.content_type.startswith("audio"):
                    content = await resp.text()
                    raise APIError(message="xtts returned non-audio data", body=content)

                # print("resp.content", resp.content)
                print("dir(resp.content)", dir(resp.content))
                
                # Prints the raw content of the API response (likely for debugging).  This should probably be removed or changed to a log.debug statement in production.
                # async for bytes_data, _ in resp.content.iter_chunks():
                #     for frame in bstream.write(bytes_data):
                #         self._event_ch.send_nowait(
                #             tts.SynthesizedAudio(
                #                 request_id=request_id,
                #                 segment_id=request_id,
                #                 frame=frame,
                #             )
                #         )
                # # Iterates through chunks of data received from the API, writes the data to the audio byte stream, and emits synthesized audio events.
                # # Each event contains the request ID, a segment ID (currently the same as the request ID), and an audio frame.

                # for frame in bstream.flush():
                #     self._event_ch.send_nowait(
                #         tts.SynthesizedAudio(request_id=request_id, frame=frame)
                #     )
                # After processing all chunks, flushes the audio byte stream to ensure all buffered data is emitted as synthesized audio events.
                output_emitter.initialize(
                    request_id=request_id,
                    sample_rate=self._opts.sample_rate,
                    num_channels=1,
                    mime_type=f"audio/{_encoding_from_format(self._opts.encoding)}",
                )

                async for data, _ in resp.content.iter_chunks():
                    print("data: ",data)
                    output_emitter.push(data)

                output_emitter.flush()

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
        # Handles potential exceptions during the API call, raising specific exceptions (APITimeoutError, APIStatusError, APIConnectionError) based on the error type.


class SynthesizeStream(tts.SynthesizeStream):
    """Streamed API using websockets"""
    # This class handles streaming synthesis using websockets.

    def __init__(
        self,
        tts: TTS,
        session: aiohttp.ClientSession,
        opts: _TTSOptions,
    ):
        # Initializes the SynthesizeStream object.
        super().__init__(tts)
        # Calls the constructor of the parent class (tts.SynthesizeStream) with the TTS object.
        self._opts, self._session = opts, session
        # Stores the TTS options and the aiohttp client session as private attributes.
        self._mp3_decoder = utils.codecs.Mp3StreamDecoder()
        # Initializes an MP3 stream decoder for MP3 encoded audio.

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        # Main asynchronous task for handling the streaming synthesis.
        self._segments_ch = utils.aio.Chan[tokenize.WordStream]()
        # Creates a channel to communicate word streams between the tokenizer and the synthesis process.

        @utils.log_exceptions(logger=logger)
        async def _tokenize_input():
            """tokenize text from the input_ch to words"""
            # Asynchronous function to tokenize the input text into words.
            word_stream = None
            async for input in self._input_ch:
                # Iterates through the input channel, processing each input.
                if isinstance(input, str):
                    # If the input is a string (text).
                    if word_stream is None:
                        # If no word stream exists, it means a new segment is starting.
                        # new segment (after flush for e.g)
                        word_stream = self._opts.word_tokenizer.stream()
                        # Creates a new word stream using the configured tokenizer.
                        self._segments_ch.send_nowait(word_stream)
                        # Sends the new word stream to the segments channel.

                    word_stream.push_text(input)
                    # Pushes the input text to the current word stream for tokenization.
                elif isinstance(input, self._FlushSentinel):
                    # If the input is a flush signal.
                    if word_stream is not None:
                        word_stream.end_input()
                        # Ends the current word stream, signaling no more input for that segment.

                    word_stream = None
                    # Resets the word stream to None, preparing for a new segment.

            self._segments_ch.close()
            # Closes the segments channel after all input is processed.


        @utils.log_exceptions(logger=logger)
        async def _run():
            # Defines an inner asynchronous function `_run` to process word streams.
            async for word_stream in self._segments_ch:
                # Iterates through word streams received from the `_segments_ch` channel.
                await self._run_ws(word_stream)
                # Calls the `_run_ws` method to handle the websocket communication for each word stream.

        tasks = [
            asyncio.create_task(_tokenize_input()),
            asyncio.create_task(_run()),
        ]
        # Creates a list of asyncio tasks: one for tokenizing input and one for processing the word streams.
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.gracefully_cancel(*tasks)
        # Runs both tasks concurrently using `asyncio.gather`.
        # In the `finally` block, it ensures that all tasks are gracefully cancelled, even if an exception occurs.


    async def _run_ws(
        self,
        word_stream: tokenize.WordStream, 
        max_retry: int = 3,
    ) -> None:
        # Asynchronously handles the websocket communication for a given word stream.
        # It manages connection retries, sends and receives data, and handles errors.
        ws_conn: aiohttp.ClientWebSocketResponse | None = None
        # Initializes a variable to hold the websocket connection, initially set to None.
        for try_i in range(max_retry):
            # Initiates a loop for retrying the websocket connection, up to max_retry times.
            retry_delay = 5
            # Sets the delay between retries to 5 seconds.
            try:
                if try_i > 0:
                    await asyncio.sleep(retry_delay)
                    # If this is not the first attempt (try_i > 0), waits for the retry delay before attempting to connect again.

                ws_conn = await self._session.ws_connect(
                    _stream_url(self._opts),
                    headers={AUTHORIZATION_HEADER: self._opts.api_key},
                )
                # Attempts to establish a websocket connection using the session, stream URL, and authorization header.
                break
                # If the connection is successful, breaks out of the retry loop.
            except Exception as e:
                logger.warning(
                    f"failed to connect to 11labs, retrying in {retry_delay}s",
                    exc_info=e,
                )
                # If an exception occurs during connection (e.g., network error), logs a warning message with the error information and retry delay.

        if ws_conn is None:
            raise Exception(f"failed to connect to 11labs after {max_retry} retries")
        # If the websocket connection is still None after all retries, raises an exception indicating that the connection failed.

        request_id = utils.shortuuid()
        segment_id = utils.shortuuid()
        # Generates unique identifiers for the request and segment.

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
        # Creates the initial message packet required by the 11labs API.
        # It includes a space as text, a flag to attempt trigger generation (though this might be deprecated as per comments in the send task),
        # voice settings (with None values removed), and a generation configuration containing the chunk length schedule.
        await ws_conn.send_str(json.dumps(init_pkt))
        # Sends the initial message packet to the websocket connection as a JSON string.
        eos_sent = False
        # Initializes a flag to track whether the "end of stream" (EOS) message has been sent.

        async def send_task():
            nonlocal eos_sent
            # Defines an inner asynchronous function to handle sending data to the websocket.
            # It uses a nonlocal variable eos_sent to track if the "end of stream" message was sent.

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
            # Defines an inner asynchronous function `recv_task` to handle receiving data from the websocket.
            # It also uses `nonlocal eos_sent` to check connection closure and utilizes `audio_bstream` to write received audio data.
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=1,
            )
            # Creates an AudioByteStream to accumulate the received audio data.  The stream is configured with the sample rate and number of channels from the TTS options.

            last_frame: rtc.AudioFrame | None = None
            # Initializes a variable to store the last processed audio frame, initially set to None.  This is used to ensure a complete frame is sent.

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
            # A helper function `_send_last_frame` is defined to emit the last processed audio frame.
            # It checks if `last_frame` is not None and then sends a `SynthesizedAudio` event to the output channel `self._event_ch`.
            # The event includes the `request_id`, `segment_id`, the audio `frame`, and a boolean flag `is_final` to indicate if it's the last frame in the segment.
            # After sending, it resets `last_frame` to None, ensuring that the same frame is not sent twice.

            while True:
                msg = await ws_conn.receive()
                # Enters an infinite loop to continuously receive messages from the websocket.
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
                # Checks if the received message indicates that the websocket connection is closed or closing.
                # If the connection is closing and the "end of stream" (eos_sent) message hasn't been sent, it raises an exception, indicating an unexpected closure.
                # Otherwise, it returns from the function, terminating the receive task.

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected 11labs message type %s", msg.type)
                    continue
                # Checks if the received message type is not TEXT (which is expected for the audio data).
                # If the type is unexpected, it logs a warning message and continues to the next iteration of the loop, ignoring the current message.

                data = json.loads(msg.data)
                # Parses the JSON data from the received message.
                encoding = _encoding_from_format(self._opts.encoding)
                # Determines the audio encoding format (e.g., "mp3", "pcm") from the TTS options.
                if data.get("audio"):
                    b64data = base64.b64decode(data["audio"])
                    # If the received data contains an "audio" field, it decodes the Base64 encoded audio data.
                    if encoding == "mp3":
                        for frame in self._mp3_decoder.decode_chunk(b64data):
                            for frame in audio_bstream.write(frame.data.tobytes()):
                                _send_last_frame(segment_id=segment_id, is_final=False)
                                last_frame = frame

                    else:
                        for frame in audio_bstream.write(b64data):
                            _send_last_frame(segment_id=segment_id, is_final=False)
                            last_frame = frame
                    # Depending on the encoding, it handles the audio data differently.
                    # If the encoding is "mp3", it uses the MP3 decoder to decode the chunk of data and then writes the decoded frames into the audio byte stream.
                    # Otherwise (if the encoding is not "mp3," assumed to be "pcm"), it directly writes the decoded data to the audio byte stream.
                    # In both cases, it then iterates through the frames produced by the stream, and for each frame:
                    #   - Calls `_send_last_frame` to emit the last complete frame received (if any), with `is_final=False`.
                    #   - Updates `last_frame` with the current frame to buffer it until a complete frame is ready or the segment ends.

                elif data.get("isFinal"):
                    for frame in audio_bstream.flush():
                        _send_last_frame(segment_id=segment_id, is_final=False)
                        last_frame = frame

                    _send_last_frame(segment_id=segment_id, is_final=True)

                    pass
                # If the received data contains "isFinal", it signals the end of the current audio segment.
                # It flushes any remaining audio data from the `audio_bstream`. Similar to above, it emits all buffered frames with `is_final=False`.
                # After flushing, it calls `_send_last_frame` one last time, now with `is_final=True`, to emit the final audio frame of the segment and mark its completion.
                # The `pass` statement does nothing and is likely a placeholder.

                elif data.get("error"):
                    logger.error("11labs reported an error: %s", data["error"])
                # If the received data contains an "error" field, it logs the error message reported by 11labs.

                else:
                    logger.error("unexpected 11labs message %s", data)
                # If none of the expected fields ("audio", "isFinal", "error") are present in the received data, it logs an error message indicating an unexpected message format.

        tasks = [
            asyncio.create_task(send_task()),
            asyncio.create_task(recv_task()),
        ]
        # Creates two asynchronous tasks: one for sending data (`send_task`) and one for receiving data (`recv_task`).

        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.gracefully_cancel(*tasks)
        # Runs both tasks concurrently using `asyncio.gather`.  The `try...finally` block ensures that both tasks are gracefully cancelled, even if an exception occurs in either of them.


def _dict_to_voices_list(data: dict[str, Any]):
    # Initializes an empty list to store Voice objects.
    voices: List[Voice] = []
    # Iterates through the list of voices in the input data.
    for voice in data["voices"]:
        # Creates a Voice object from the current voice data and appends it to the voices list.
        # The Voice object is created using the voice's ID, name, and category from the input data.
        # Settings are set to None, indicating no specific voice settings are applied.
        voices.append(
            Voice(
                id=voice["voice_id"],
                name=voice["name"],
                category=voice["category"],
                settings=None,
            )
        )
    # Returns the list of Voice objects.
    return voices

def _strip_nones(data: dict[str, Any]):
    # Returns a new dictionary containing only the key-value pairs from the input dictionary
    # where the value is not None.  This effectively removes entries with None values.
    return {k: v for k, v in data.items() if v is not None}


def _synthesize_url(opts: _TTSOptions,text:str) -> str:
    # Imports the urllib.parse module for URL encoding.
    import urllib.parse
    # Retrieves the default speaker from the environment variable "XTTS_SPEAKER", defaulting to "male.wav" if not set.
    default_speaker = os.environ.get("XTTS_SPEAKER", "male.wav")
    # Retrieves the default language from the environment variable "XTTS_LANGUAGE", defaulting to "en" if not set.
    default_language = os.environ.get("XTTS_LANGUAGE", "en")
    # URL-encodes the input text to ensure it's safe for use in a URL.
    safe_string = urllib.parse.quote(text)
    # Constructs the base URL for the synthesis request, including the text, speaker, and language as query parameters.
    base_url = f"{opts.base_url}/tts_stream?text={safe_string}&speaker_wav={default_speaker}&language={default_language}"
    # Returns the constructed base URL.
    return (
        f"{base_url}"
    )


def _stream_url(opts: _TTSOptions) -> str:
    # Extracts the base URL from the TTS options.
    base_url = opts.base_url
    # Extracts the voice ID from the TTS options.
    voice_id = opts.voice.id
    # Extracts the model ID from the TTS options.
    model_id = opts.model
    # Extracts the desired output format (encoding) from the TTS options.
    output_format = opts.encoding
    # Extracts the streaming latency setting from the TTS options.
    latency = opts.streaming_latency
    # Converts the enable_ssml_parsing boolean to a lowercase string ("true" or "false").
    enable_ssml = str(opts.enable_ssml_parsing).lower()
    # Extracts the language code from the TTS options.
    language = opts.language
    # Constructs the base URL for the streaming request, including voice ID and various query parameters.
    url = (
        f"{base_url}/text-to-speech/{voice_id}/stream-input?"
        f"model_id={model_id}&output_format={output_format}&optimize_streaming_latency={latency}&"
        f"enable_ssml_parsing={enable_ssml}"
    )
    # If a language code is provided, appends it as a query parameter to the URL.
    if language is not None:
        url += f"&language_code={language}"
    # Returns the complete constructed URL for the streaming request.
    return url

