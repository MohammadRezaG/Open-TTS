# Streaming, speech speed, and sentence pauses

Status: streaming and pause controls implemented. The post-processing speed control described in this historical plan has been removed (2026-10-14) because the user wants the model itself to speak slower. Pocket TTS currently exposes no native speech-rate option; model-native rate control remains open work.

## Intended behavior

Playback starts while speech is still being generated. The user controls a multiplier for the automatic silent pause between sentences. Speech rate must come from a model that supports or has been trained for rate control; post-processing the generated audio is not desired. Here, breathing space means a silent pause; adding an audible inhale would be a separate feature.

The automatic pause uses punctuation and the lengths of both neighboring sentences. The multiplier scales the entire calculation, including both length contributions. Speed changes spoken audio while the pause multiplier controls the intentional gaps.

## Findings from the local code

- `engine/src/open_tts_engine/pocket_tts.py` calls `generate_audio()`, collects every text chunk, inserts a fixed 250 ms silence between chunks, and finally creates one WAV.
- `api/src/open_tts_api/main.py` returns that completed WAV. `webui/src/open_tts_webui/app.py` waits for the complete HTTP response before giving a file to the player. Setting autoplay alone cannot stream this path.
- `ref-project/webui-v2/farsi_webui.py` calls `generate_audio_stream()`, yields roughly 500 ms audio batches, and uses a streaming Gradio audio component. It also applies 12 ms boundary fades. Its inserted silence is still a fixed 250 ms.
- The installed Pocket TTS 3.3.0 exposes `generate_audio_stream(..., stop=threading.Event())`. Its generation and decoding use worker threads. It has no speech-speed parameter in this method.
- Our current `frames_after_eos=options.frames_after_eos or None` maps a requested zero to the provider's automatic setting. Correct this so zero means zero; keep end-of-speech frames separate from the pause multiplier and listen for clipped endings when reducing them.
- The installed Gradio 6.29.0 supports streaming audio bytes. Its NumPy output path normalizes each supplied array, so use WAV byte blocks to avoid changing volume independently for each block.
- FFmpeg is available at `D:\ffmpeg\bin\ffmpeg.exe`. Reuse its tempo filter for speed processing instead of adding another Python audio library.
- The present text chunker keeps punctuation but returns only strings. It needs sentence/boundary information so a size-limit split does not become an artificial sentence pause.

These are source observations. No playback timings or audio measurements were taken for this plan. The fixed silence explains one source of equal gaps; the model's own trailing silence and streaming buffer underruns also need measurement.

## Implementation order

### 1. Stream through the shared engine, API, and UI

Add an engine iterator yielding sample rate and incremental audio samples. Normalize and split the input first, but phonemize each chunk just before it is needed. Begin generating and yielding the first chunk without waiting for later chunks. Keep the text-versus-phoneme input modes intact.

Make the existing WAV method collect this same iterator, so streaming and file generation share their processing rules.

Add `POST /v1/speech/stream` using FastAPI `StreamingResponse`. Use a readable JSON Lines stream: an initial sample-rate/PCM-format record, audio records containing base64 PCM16 samples, and an explicit completion or error record. The UI reads lines incrementally with `httpx.stream`; HTTP packet boundaries must not be treated as audio boundaries. Keep the existing WAV endpoint for callers requesting a complete file.

Turn the UI callback into a generator and enable the streaming audio component. Batch around 0.5 seconds initially, matching the reference's starting point, and flush a shorter final block. Yield self-contained WAV byte blocks to Gradio. Save the received PCM to a final downloadable WAV during that same request, without generating the speech again. Preserve the current design and English labels.

Retain the engine generation lock for a stream's lifetime. Clean up the voice recording, iterator, lock, and any audio processor on completion, failure, or cancellation. Add a Stop control that closes the stream and signals Pocket TTS's stop event. Retry a failed chunk only before any of its audio has reached playback; replaying a partially emitted chunk would duplicate speech.

### 2. Replace fixed silence with an automatic pause

Retain each chunk's logical sentence identity, normalized sentence word count, and boundary type before G2P discards punctuation. Use the full neighboring sentence lengths at a sentence boundary, even when either sentence was split into several generation chunks.

Start with this small, deterministic heuristic:

```text
automatic_pause = min(
    0.50 seconds,
    punctuation_base
    + 0.004 seconds * min(previous_sentence_words, 30)
    + 0.003 seconds * min(next_sentence_words, 30)
)

final_pause = pause_multiplier * automatic_pause
```

Provisional punctuation bases:

| Boundary | Base | Length contributions |
| --- | ---: | --- |
| Period | 120 ms | Both neighboring sentences |
| Question/exclamation | 160 ms | Both neighboring sentences |
| Semicolon | 90 ms | Both neighboring phrases |
| Comma | 50 ms | Half the weights, using adjacent phrases |
| Character-limit split within a phrase | 0 ms | None |
| End of the complete input | 0 ms added | None |

For example, a period between a 3-word and a 4-word sentence gives 144 ms at 1x, 72 ms at 0.5x, and 288 ms at 2x. A 20-word sentence followed by a 25-word sentence gives 275 ms at 1x. These are initial listening values, not claims of linguistic precision.

Insert audio samples for the pause; do not sleep in the generation code. Use short boundary fades. Conservatively remove excessive leading/trailing near-silence at generation-chunk edges, retaining a small guard so quiet consonants are preserved. Account for retained edge silence when adding the gap rather than stacking the entire requested gap on top of it. A small tail buffer allows this without holding an entire sentence.

Log the intended pause and retained edge silence. Distinguish the pause encoded in the audio from an extra playback stall caused by the next chunk not being ready. If measurements show stalls, add bounded preparation of the next chunk's G2P while current audio is playing; do not run concurrent Pocket TTS generations on the same model.

### 3. Add the two controls

| English UI label | API/engine field | Initial range | Default |
| --- | --- | --- | --- |
| Speech speed | `speed` | 0.75–1.50x, step 0.05 | 1.00x |
| Breath / pause length | `pause_multiplier` | 0.00–2.00x, step 0.10 | 1.00x |

Explain below the pause slider: "Scales the automatic pause based on both neighboring sentences."

Validate both values in the API and engine. Apply controls to streaming playback and the downloadable WAV. Values are captured when Generate is pressed; changing a slider affects the next request.

For non-default speed, stream PCM through FFmpeg `atempo` with pitch preservation as an acceptance requirement. Keep the processor alive through the request rather than launching a process for every small audio frame. At 1x, bypass tempo processing entirely. Since tempo processing also scales silence, compensate inserted silence by the speed factor before the filter so its final duration follows the pause calculation; allow a small tolerance for filter windows. Drain output concurrently with input and flush the filter at completion. Bound queues and terminate the process on cancellation.

Measure filter startup latency and boundary quality before accepting this path. Do not buffer the whole utterance or silently change pitch to implement speed. If FFmpeg is unavailable, return a clear error for non-default speed while the 1x path remains usable.

### 4. Check the actual runtime and audio

Before implementation checks, confirm which Python environment the running API/UI use and install new requirements there if needed. Do not assume an older environment warning still describes the user's running setup.

Measure warm-model request-to-first-engine-frame, request-to-first-HTTP-audio, request-to-audible-browser-playback, total duration, and playback underruns. Compare our version with ref 1 using the same text, voice, model, and hardware. Report cold startup separately.

Required checks after implementation:

- Streaming emits audio before the last sentence is generated, rather than returning a finished file in a streaming wrapper.
- API validation, upload handling, cancellation, mid-stream errors, and the next request after cancellation work correctly.
- Pause calculations use both neighboring sentences; the multiplier scales both contributions. Size-limit splits add no sentence pause, and the last chunk adds no artificial final gap.
- At 0.5x/1x/2x pause settings, measured inserted gaps follow their targets within the audio filter's documented tolerance.
- At 0.75x/1x/1.5x speech speed, spoken duration changes as expected, pitch remains consistent, and intentional pauses still follow their own multiplier.
- Stream parsing handles HTTP reads that split or combine JSON Lines records.
- Quiet word endings remain intact; no clicks, duplicated speech, missing speech, or per-block volume changes appear.
- The downloaded WAV contains the same processed speech and pauses as playback. Model loading and both input modes still follow the existing provider configuration.

Listening examples:

```text
سلام. حالت چطور است؟ من خوبم.

من امروز برای خرید چند کتاب و بررسی قیمت آن‌ها به فروشگاه نزدیک خانه رفتم. تو چه خریدی؟

من خوبم. امروز می‌خواهم درباره موجودی کالاها، قیمت خرید و برنامه تحویل سفارش‌های هفته آینده صحبت کنم.

اول قیمت را بررسی کن، بعد موجودی را ببین؛ سپس سفارش را ثبت کن.

قیمت 1,250 تومان است. شماره من 0912 است. مقدار ۳٫۵ است.
```

Use a longer paragraph with a low maximum chunk length to exercise forced splits. Repeat tests with both neighboring sentence lengths changed independently.

## Files expected to change during implementation

- `engine/src/open_tts_engine/contract.py`: incremental speech contract.
- `engine/src/open_tts_engine/options.py`: speed and pause multiplier.
- `engine/src/open_tts_engine/chunking.py`: sentence and boundary information.
- `engine/src/open_tts_engine/pocket_tts.py`: incremental model frames and shared stream/file path.
- One small engine audio-processing module: fades, edge-silence handling, pause calculation, and tempo processing.
- `api/src/open_tts_api/main.py`: streaming endpoint, controls, validation, resource lifetime.
- `webui/src/open_tts_webui/app.py`: incremental API consumption, streaming player, two sliders, stop, and download.
- Focused engine/API tests and README instructions for streaming and FFmpeg.

The existing uncommitted changes, including the UI port setting, must be preserved. The reference projects remain comparison material.

## Verification status

- Passed the focused engine assertions for normalization, pause scaling, artificial split boundaries, silence trimming, WAV output, the zero end-frame setting, and FFmpeg duration and pitch behavior on a synthetic tone.
- Passed API tests for PCM stream records and cancellation, plus a UI test that confirms the streamed WAV block and final download contain the same samples. The Gradio interface builds successfully.
- Real Pocket TTS and G2P generation, browser playback latency, and listening quality have not been measured. Those still require restarting the local API/UI and listening to the sample paragraphs above.

## Sources

- Local reference: `ref-project/webui-v2/farsi_webui.py`, especially `_generate_phonemes`, `_stream_phonemes_faded`, and `synthesize_streaming`.
- Installed provider: `.venv/Lib/site-packages/pocket_tts/models/tts_model.py`.
- Installed player: `.venv/Lib/site-packages/gradio/components/audio.py`.
- [Gradio streaming generated audio](https://www.gradio.app/guides/streaming-ai-generated-audio).
- [FastAPI StreamingResponse](https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse).
- [FFmpeg atempo](https://ffmpeg.org/ffmpeg-filters.html#atempo).
