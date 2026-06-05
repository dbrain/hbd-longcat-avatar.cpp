# NAVA cpp — Audio-VAE ENCODER port spec (ground-truth, ready to implement)

Branch `nava-port`. DEFERRED (owner: do after the deep perf run). This doc is a COMPLETE
blueprint built from the actual checkpoint tensor shapes — no remaining architecture
ambiguity. Implement, validate vs python `wrapped_encode`, then wire external-wav -> audio
condition for "shared audio across the whole clip" (multi-segment non-restarting speech).

## Why: the audio encoder = the ONLY missing piece for shared external-audio drive
Clip chaining (Track A) needs no encoder. Driving the whole clip from ONE external waveform
does: wav -> mel -> encoder -> audio latent [128,T] -> use as the audio stream / per-segment
audio condition (the per-token-t=0 `--audio-anchor` plumbing already exists in main.cpp).

## GROUND-TRUTH architecture (from ltx-2.3-22b-dev_audio_vae.safetensors header)
Encoder = 46 tensors (44 encoder + 2 stats). **3 levels, NO attention anywhere, PixelNorm
(parameter-free) — much simpler than the LTX default signature suggests.** All conv weights
are the SAME building blocks already ported on the DECODE side (`src/ltx_audio_vae.h`).

```
input mel  [B, 2, T_mel, 64]            (2 audio channels, 64 mel bins; T axis=dim2, F axis=dim3)
conv_in    CausalConv2d 2->128  k3 s1   WIDTH-causal     weight [128,2,3,3]
down.0:    ResnetBlock 128->128 (x2)                     conv1/conv2 [128,128,3,3]
           downsample CausalConv2d 128->128 k3 s2 WIDTH  [128,128,3,3]   (halves T and F)
down.1:    ResnetBlock 128->256 (nin_shortcut [256,128,1,1]) + ResnetBlock 256->256
           downsample CausalConv2d 256->256 k3 s2 WIDTH  [256,256,3,3]   (halves T and F)
down.2:    ResnetBlock 256->512 (nin_shortcut [512,256,1,1]) + ResnetBlock 512->512
           NO downsample (last level)
mid:       block_1 ResnetBlock 512->512 ; block_2 ResnetBlock 512->512   (NO attn_1 — absent)
norm_out:  PixelNorm2D (no params) ; SiLU
conv_out:  CausalConv2d 512->16  k3 s1  WIDTH            weight [16,512,3,3]   (16 = 2*z, z=8)
-> chunk(2,dim=channels)[0]        keep first 8 channels  -> [B,8,T/4,16]
-> patchify "b c t f -> b t (c f)"  -> [B, T/4, 128]
-> per_channel normalize (x - mean-of-means)/std-of-means   stats [128]
-> unpatchify -> [B,8,T/4,16] ; wrapped_encode then patchify+transpose -> [B,128,T/4]
```
Net: T downsampled by 4 (latent_downsample_factor), F 64->16. Output audio latent **[128, T/4]**
= the exact shape the DiT consumes (`audio_latent` in main.cpp, [128, audio_len]).

Causality: encoder is **WIDTH-causal** (pad `(pad_w,0,pad_h//2,pad_h-pad_h//2)`; k3 -> (2,0,1,1)).
Downsample causal conv pad = `(2,0,0,1)` (left=2,right=0,top=0,bot=1), stride 2.
NB: decode side `HeightCausalConv2D` is HEIGHT-causal — the encoder needs a WIDTH variant
(swap which axis gets the asymmetric pad). Verify the T/F axis mapping against ggml ne order
([W,H,...] = ne0,ne1) when porting — torch [B,C,H,W] H=dim2=T, W=dim3=F.

### Tensor name tree (verbatim gguf keys after un-skip; loader does NO axis reversal)
```
audio_vae.encoder.conv_in.conv.{weight,bias}
audio_vae.encoder.down.{0,1,2}.block.{0,1}.conv1.conv.{weight,bias}
audio_vae.encoder.down.{0,1,2}.block.{0,1}.conv2.conv.{weight,bias}
audio_vae.encoder.down.{1,2}.block.0.nin_shortcut.conv.{weight,bias}   (only where ch changes)
audio_vae.encoder.down.{0,1}.downsample.conv.{weight,bias}
audio_vae.encoder.mid.block_{1,2}.conv{1,2}.conv.{weight,bias}
audio_vae.encoder.conv_out.conv.{weight,bias}
audio_vae.per_channel_statistics.{mean-of-means,std-of-means}   (already in the gguf, [128])
```
ResnetBlock has NO norm tensors (PixelNorm2D = parameter-free RMS-style). temb_channels=0.

## MEL FRONT-END (the fiddly bit — torchaudio MelSpectrogram, exact)
`AudioProcessor.waveform_to_mel` (nava_src/.../audio_vae/ops.py:44-55):
- torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=1024, win_length=1024,
  hop_length=160, f_min=0, f_max=8000, n_mels=64, window=hann, center=True, pad_mode="reflect",
  power=1.0 (MAGNITUDE not power), mel_scale="slaney", norm="slaney")
- then `log(clamp(mel, min=1e-5))`, then permute to [B, C, T, 64].
- channels adapted to 2 BEFORE mel (mono->2). wrapped_encode input: waveform [C,samples] or
  [B,C,samples] @ 16k (resample if needed).

**Recommended cpp approach** (avoid reimplementing slaney in cpp): precompute in the converter
and bake into the gguf, mirroring the existing decode-side BWE mel:
- `forward_basis` [n_fft=1024, 1, (n_fft/2+1)*2] = windowed DFT (Hann folded in), real+imag
  stacked — same layout the decode `compute_log_mel_spectrogram` expects as a conv1d kernel.
- `mel_basis` [n_fft/2+1=513, 64] = torchaudio slaney filterbank (f_max=8000, norm=slaney).
Then cpp reuses `compute_log_mel_spectrogram(...)` BUT note its padding is LEFT-pad
(`filter_len-hop`), whereas torchaudio center=True reflect-pads `n_fft/2` on BOTH sides. For
bit-parity you must add center reflect padding (n_fft/2 each side) before the conv1d, OR accept
a small edge mismatch (validate the interior frames). Build `mel_basis` with
`torchaudio.functional.melscale_fbanks(..., norm="slaney", mel_scale="slaney")` to match exactly.

## CONVERTER change (tools/convert_ltx_audio_vae.py)
- Remove/relax `SKIP_PREFIX = "audio_vae.encoder"` so the 44 encoder tensors pack (verbatim
  names, dtype policy: conv weights F16, biases F32 — same as decode).
- Add `audio_vae.encoder.mel_stft.forward_basis` (F32) + `...mel_basis` (F32) computed as above
  (new keys; the encoder cpp loads them under its own prefix).
- Repack:
  `/mnt/hdd/nava/.venv/bin/python tools/convert_ltx_audio_vae.py --src /mnt/hdd/nava/params/LTX2/ltx-2.3-22b-dev_audio_vae.safetensors --out models/nava-ltx-audio-vae-f16.gguf`
  (two-step: NAVA venv dumps npz, re-execs `uv run --with gguf` to pack — automatic.)
  Source safetensors: `/mnt/hdd/nava/params/LTX2/ltx-2.3-22b-dev_audio_vae.safetensors` (348 MB).

## cpp implementation plan (src/ltx_audio_vae.h)
Reuse: `PixelNorm2D` (472), `AudioResnetBlock2D` (530), the conv wrappers. NEW: a WIDTH-causal
conv (parameterize `HeightCausalConv2D` by axis, or add `WidthCausalConv2D`), a strided
`AudioDownsample2D` (causal conv stride2). Build `struct AudioEncoder` mirroring the decode
`AudioDecoder`. Add `LTXAudioVAERunner::encode(n_threads, waveform)` -> latent [128,T/4].
Add `nava audio-encode <gguf> <wav> <out_latent.bin>` subcommand (mirror `ltx-audio-test`).

## VALIDATION
- Python ref: `python -c "from nava_src.vae.local_audio_vae import ...; wrapped_encode(wav)"`
  (or write a tiny `nava_audio_vae_encode_ref.py` next to the decode ref). Dump latent.
- Gate: cpp encode() vs python latent — correlation/cos per channel. Mel parity first (compare
  the mel tensor), then the full encode. (Latent cosine OVERSTATES the perceptual gap for audio
  — see Track A perf notes — so also round-trip encode->decode->wav and judge by ear.)

## WIRING (after validation)
External wav -> encode -> audio latent [128, T_total]. For multi-segment: slice the latent per
segment (audio_len tokens each, offset = seg*new_per_seg) and feed as the audio stream
(`--audio-anchor` clean-pin, or as the init audio_latent via NAVA_INJECT_AUDIO). One shared
track => continuous speech across the whole chained clip (pairs with M=1 video continuation).
