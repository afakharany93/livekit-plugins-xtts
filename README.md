# The original LiveKit Plugins XTTS was by Anish Menon. I just edited it to work with version 1.1.3 of livekit agents 

## Pre-requisites

The following environment variables need to be set:

| Variable | Description | Example |
|----------|-------------|---------|
| `XTTS_BASE_URL` | Base URL of the XTTS server | `http://localhost:8000` |
| `XTTS_SPEAKER` | Path to the speaker file as per the XTTS server repo | `speaker.wav` |
| `XTTS_LANGUAGE` | Language code for TTS | `en` |

## Installation

### Step 1: Install the XTTS Server (open source)

Follow the instructions to setup the XTTS Server:
https://github.com/daswer123/xtts-api-server

### Step 2: Install the LiveKit Plugin

```bash
pip install git+https://github.com/afakharany93/livekit-plugins-xtts.git

```

## Usage examples folder

```

```


