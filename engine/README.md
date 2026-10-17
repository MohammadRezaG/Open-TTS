# Engine

Owns Persian text cleanup and G2P, language chunking, model adapters, streamed
PCM, speed processing, and WAV generation. The API consumes the same stream as
the WAV writer. `LanguageRoutedEngine` sends Persian to Pocket TTS and English
to Kitten Mini. The Persian G2P formatting remains specific to Pocket.

Sentence pauses use punctuation and both neighboring phrase lengths. Kitten's
native speed parameter and optional FFmpeg pitch-preserving tempo processing
are independent; punctuation pauses are inserted after speech processing.

Installed models load from local assets. Pocket runs with PyTorch on CPU/CUDA;
Kitten runs with ONNX Runtime on CPU/CUDA when the provider is installed.
`GenerationOptions` carries language, named voice, native speed, post speed,
and independent pause settings.
