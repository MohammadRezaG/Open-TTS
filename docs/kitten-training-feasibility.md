# Kitten Mini 0.8 training feasibility

Checked 2026-10-15 against the public KittenTTS repository and the Mini 0.8
model repository.

## Finding

The public Mini 0.8 repository provides an ONNX inference model, `config.json`,
and `voices.npz`. The upstream Python interface documents model loading,
generation, named voices, speed, and CPU/GPU inference. It does not provide a
Mini 0.8 trainable checkpoint, training source, or fine-tuning recipe.

This project therefore does not create a Kitten trainer or convert the ONNX
weights into an assumed trainable format. Fine-tuning Kitten Mini for one
bilingual English-Persian voice is **not currently established as feasible from
the published Mini 0.8 assets**. Revisit only if Kitten publishes the training
code/checkpoint or provides a compatible conversion and recipe. A different
architecture would need a separate decision.

## Data candidates

- [MirasVoice](https://github.com/miras-tech/MirasVoice) describes English and
  Persian recordings from the same bilingual speakers, which matches the
  paired-speaker goal. The repository says access to the complete corpus and
  text materials requires a request. Confirm the data license, training rights,
  and access terms before using any data.
- [FLEURS](https://huggingface.co/datasets/google/fleurs) includes Persian
  (`fa_ir`) speech and is published under CC-BY-4.0. It can help validate
  Persian audio/transcript preparation, but it does not provide the same
  English-Persian speakers needed for a single bilingual voice.

## Next gate

Before building any trainer, obtain a documented trainable base/checkpoint and
an authorized same-speaker bilingual dataset. Then confirm sample rate,
transcript alignment, speaker labels, and train/dev/test partitioning against
the model's actual recipe.
