#!/usr/bin/env python3
"""Dump intermediate tensors from the Python MOSS-Transcribe-Diarize reference.

Captures, for a given WAV, every node the Rust port reproduces:
  - mel input_features (post WhisperFeatureExtractor)
  - whisper encoder last_hidden_state (per chunk)
  - per-audio trimmed+concat features
  - time-merged features
  - VQAdaptor output (audio embeds)
  - input_ids, attention_mask, audio_feature_lengths, audio_chunk_mapping
  - generated token ids
  - final text

Run in the moss-transcribe-diarize conda env:
  python rust/scripts/dump_reference.py 15s.wav --out rust/_ref/15s
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from moss_transcribe_diarize.inference_utils import (
    build_transcription_messages,
    generate_transcription,
    load_audio_item,
)
from moss_transcribe_diarize.processing_moss_transcribe_diarize import (
    _audios_to_input_features,
    _chunk_audio,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--out", default=None, help="output dir (default rust/_ref/<audio-stem>)")
    ap.add_argument("--model", default="pretrained/moss-transcribe-diarize")
    ap.add_argument("--prompt", default="Transcribe the audio as text.")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    audio_path = args.audio
    stem = Path(audio_path).stem
    out_dir = Path(args.out) if args.out else Path("rust/_ref") / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading model from {args.model} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, dtype="auto").to("cuda").eval()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, fix_mistral_regex=True)

    sr = processor.feature_extractor.sampling_rate
    audio = load_audio_item(audio_path, sr)
    np.save(out_dir / "audio.npy", audio)

    # --- processor-level intermediates ---
    feat_ext = processor.feature_extractor
    chunks, token_lengths = _chunk_audio(feat_ext, audio, processor.audio_merge_size)
    np.save(out_dir / "chunks.npy", np.stack(chunks))
    np.save(out_dir / "token_lengths.npy", np.array(token_lengths, dtype=np.int64))

    input_features, audio_feature_lengths, audio_chunk_mapping = _audios_to_input_features(
        feat_ext, [audio], audio_merge_size=processor.audio_merge_size,
    )
    np.save(out_dir / "input_features.npy", input_features.cpu().numpy())
    np.save(out_dir / "audio_feature_lengths.npy", audio_feature_lengths.cpu().numpy())
    np.save(out_dir / "audio_chunk_mapping.npy", audio_chunk_mapping.cpu().numpy())

    # --- whisper encoder output (audio embeds before scatter) ---
    with torch.no_grad():
        audio_embeds = model.get_audio_features(
            input_features=input_features.to("cuda"),
            audio_feature_lengths=audio_feature_lengths.to("cuda"),
            audio_chunk_mapping=audio_chunk_mapping.to("cuda"),
        )
    # audio_embeds is a list of (1, N, hidden) tensors per audio; concat.
    ae = torch.cat([f.squeeze(0) for f in audio_embeds], dim=0).float().cpu().numpy()
    np.save(out_dir / "audio_embeds.npy", ae)
    print(f"audio_embeds shape: {ae.shape}", flush=True)

    # --- chat-template prompt + input_ids ---
    messages = [{"role": "user", "content": [
        {"type": "audio", "audio": audio_path},
        {"type": "text", "text": args.prompt},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = processor(text=text, audio=[audio], max_length=131072, return_tensors="pt")
    input_ids = enc["input_ids"][0].cpu().numpy()
    np.save(out_dir / "input_ids.npy", input_ids)
    np.save(out_dir / "attention_mask.npy", enc["attention_mask"][0].cpu().numpy())

    prompt_len = int(enc["attention_mask"][0].sum().item())
    print(f"prompt_len={prompt_len}", flush=True)

    # --- generation (greedy) — capture token ids directly ---
    with torch.no_grad():
        outputs = model.generate(
            input_ids=enc["input_ids"].to("cuda"),
            attention_mask=enc["attention_mask"].to("cuda"),
            input_features=enc["input_features"].to("cuda"),
            audio_feature_lengths=enc["audio_feature_lengths"].to("cuda"),
            audio_chunk_mapping=enc["audio_chunk_mapping"].to("cuda"),
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    gen_ids = outputs[0][prompt_len:].cpu().numpy()
    np.save(out_dir / "generated_ids.npy", gen_ids.astype(np.int64))
    text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    res = {"text": text, "prompt_len": prompt_len, "generated_tokens": int(len(gen_ids))}
    with open(out_dir / "text.txt", "w", encoding="utf-8") as f:
        f.write(res["text"])
    with open(out_dir / "meta.txt", "w", encoding="utf-8") as f:
        f.write(f"prompt_len={res['prompt_len']}\n")
        f.write(f"generated_tokens={res['generated_tokens']}\n")
        f.write(f"audio_token_id={processor.audio_token_id}\n")
        f.write(f"audio_merge_size={processor.audio_merge_size}\n")
        f.write(f"time_marker_every_seconds={processor.time_marker_every_seconds}\n")
        f.write(f"audio_tokens_per_second={processor.audio_tokens_per_second}\n")
        f.write(f"digit_token_ids={processor.digit_token_ids}\n")
    print(f"wrote {out_dir}", flush=True)
    print(f"text: {res['text'][:200]}", flush=True)


if __name__ == "__main__":
    main()
