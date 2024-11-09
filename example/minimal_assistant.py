import asyncio
import logging

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
    metrics,
    stt,
    transcription,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, openai, silero, xtts

load_dotenv()
import os
logger = logging.getLogger("voice-assistant")

async def _forward_transcription(
    stt_stream: stt.SpeechStream, stt_forwarder: transcription.STTSegmentsForwarder
):
    """Forward the transcription to the client and log the transcript in the console"""
    async for ev in stt_stream:
        stt_forwarder.update(ev)
        if ev.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
            print(f"INTERIM: {ev.alternatives[0].text}", end="")
        elif ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
            print("\nFINAL ->", ev.alternatives[0].text)

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are a voice assistant created by LiveKit. Your interface with users will be voice. "
            "You should use short and concise responses, and avoiding usage of unpronouncable punctuation."
        ),
    )

    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)

    async def transcribe_track(participant: rtc.RemoteParticipant, track: rtc.Track):
        audio_stream = rtc.AudioStream(track)
        stt_forwarder = transcription.STTSegmentsForwarder(
            room=ctx.room, participant=participant, track=track
        )
        stt_stream = openai.STT(
            model=os.getenv("WHISPER_MODEL"),
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY")
        ).stream()
        asyncio.create_task(_forward_transcription(stt_stream, stt_forwarder))

        async for ev in audio_stream:
            stt_stream.push_frame(ev.frame)

    async def participant_task(participant: rtc.RemoteParticipant):
        logger.info(f"starting voice assistant for participant {participant.identity}")

        # Handle existing tracks
        for publication in participant.track_publications.values():
            track = publication.track
            if track and not track.subscribed:
                await track.subscribe()
                if track.kind == rtc.TrackKind.KIND_AUDIO:
                    asyncio.create_task(transcribe_track(participant, track))

        # Handle track subscriptions
        @ctx.room.on("track_subscribed")
        def on_track_subscribed(track: rtc.Track, publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                asyncio.create_task(transcribe_track(participant, track))

        agent = VoicePipelineAgent(
            vad=ctx.proc.userdata["vad"],
            stt=openai.STT(model=os.getenv("WHISPER_MODEL"),base_url="https://api.groq.com/openai/v1",api_key="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"),
            llm=openai.LLM(
                model="llama3.2",
                base_url="http://localhost:11434/v1",
                api_key="not-needed"
            ),
            tts=xtts.TTS(
                api_key="dummy",
                voice=xtts.Voice(
                    id="v2",
                    name="default",
                    category="neural",
                    settings=None
                )),
            chat_ctx=initial_ctx,
        )

        agent.start(ctx.room, participant)
        await agent.say("Hey, how can I help you today?", allow_interruptions=True)

        # Keep the participant task running
        while True:
            await asyncio.sleep(1)

    # Handle existing participants
    for participant in ctx.room.remote_participants.values():
        asyncio.create_task(participant_task(participant))

    # Handle new participants
    @ctx.room.on("participant_connected")
    async def on_participant_connected(participant):
        await participant_task(participant)

    # Keep the connection alive
    while True:
        await asyncio.sleep(1)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
