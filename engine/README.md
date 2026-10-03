# Engine

Owns Persian normalization and phonemization, Pocket TTS loading, voice prompts,
chunking, and WAV generation. `PocketTtsEngine` is the provider implementation;
`SpeechEngine` is the small interface the API depends on.

The engine loads the Farsi v2 model and G2P model once when the API starts. Model
IDs, temperature, and default voice can be changed with `OPEN_TTS_TTS_CONFIG`,
`OPEN_TTS_G2P_MODEL`, `OPEN_TTS_TEMPERATURE`, and `OPEN_TTS_DEFAULT_VOICE`.