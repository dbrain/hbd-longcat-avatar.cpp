# R4 — SkinTokens skin-VAE (FSQ-CVAE) architecture

Probe of `vae.*` in `experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt`
(252 tensors, all bfloat16 in-ckpt → dumped as fp32). The VAE was pretrained as
`experiments/skin_vae_2_10_32768/last.ckpt` and loaded via `pretrained_vae`; the deployed
weights used by the e2e golden are the grpo ckpt's `vae.*` (used here — the grpo `FSQ.project_out`
differs from the vae-only ckpt by ~5e-4 bf16, so always validate against the grpo copy).

Class: `SkinFSQCVAEModel` (`src/model/skin_vae/autoencoders/skin_fsq_cvae_model.py`).
The full skin path: `src/model/skin_vae_model.py::SkinVAEModel.decode()` → `model._decode()`;
the token-level driver is `src/model/tokenrig.py::decode()` (~L536).

## Real config (from VAE ckpt `hyper_parameters.model_config.model`)

```
in_channels      = 4        cond_channels    = 3
latent_channels  = 512      num_attention_heads = 12
width_encoder    = 768      width_decoder    = 768     (ff inner = 3072 = 4×768)
num_layers_encoder = 2      num_layers_decoder = 10
embedding_type   = frequency  embed_frequency = 8  embed_include_pi = TRUE  use_pmpe = TRUE
is_learned_queries = TRUE     sample_tokens/compress_tokens = 32   cond_tokens = 384
FSQ_dict = { levels: [8,8,8,8,8], dim: 512 }
```

NB: heads=12, width=768 → head_dim = 64. ff = GEGLU-style `net.0.proj` (768→3072) + `net.2`
(3072→768). LayerNorm before each sub-block (norm1/norm2/norm3, all dim 768, with weight+bias).

### Fourier embedder out_dim
`FrequencyPositionalEmbedding(num_freqs=8, include_pi=True, input_dim=3, use_pmpe=True)`.
Empirically (from proj_in/proj_query in-dims): point-position embed = 51 dims, so:
- encoder `proj_in` in = 55 = `in_channels(4) + embed.out_dim(51)`
- cond_encoder `proj_in` in = 54 = `cond_channels(3) + 51`
- decoder query `proj_query` in = 54 = `cond_channels(3) + 51`
→ **embed.out_dim = 51** for 3-D input (`3 + 3*2*8 = 51`; logspace, include_pi). Confirm the exact
PMPE formula in `embeddings.py::FrequencyPositionalEmbedding` when porting the encoder/decoder.

## FSQ (the R4 deliverable — DONE, ported in `rig_fsq.hpp`)

`src/model/skin_vae/autoencoders/FSQ.py`. levels = **[8,8,8,8,8]** (REAL; NOT [8,8,8,5,5,5]).
- `codebook_dim` = len(levels) = 5
- `codebook_size` = prod(levels) = 8^5 = **32768** == `vae_vocab`
- `_basis` = cumprod([1]+levels[:-1]) = **[1, 8, 64, 512, 4096]**
- `half_width` = levels//2 = [4,4,4,4,4]
- has_projections (dim 512 ≠ codebook_dim 5) → TRUE:
  - `project_in`  : Linear(512→5)  [encode side; ne [5,512]] — not used in decode
  - `project_out` : Linear(5→512)  [ne [512,5]] — applied by `indices_to_codes`

`indices_to_codes(idx)` = `project_out( _indices_to_codes(idx) )`:
1. per-dim digit `li = (idx // basis[d]) % levels[d]`
2. `_scale_and_shift_inverse`: `code_d = (li - half_width_d) / half_width_d`
   (levels=8 even ⇒ codes ∈ {-1,-.75,-.5,-.25,0,.25,.5,.75}, asymmetric, no +1.0)
3. Linear 5→512 with bias.
Output `[N, 512]`. C++ port is **bit-exact** (max-abs diff 0.0 vs Python, both raw and post-project).

## Encoder / cond_encoder (`Tripo2Encoder`, 2 layers → 3 blocks each)

Per-group tensors (45–46 each):
- `proj_in` : Linear(in → 768)  (in = 55 enc / 54 cond)
- `learned_queries` : [32, 768]  **(encoder only; is_learned_queries=True)** — the 32 latent
  query tokens. cond_encoder has NO learned_queries (it queries from FPS-sampled cond points).
- `blocks.0` : **cross-attn (attn2)** + ff  — queries ← (learned_queries / sampled cond);
  KV ← the embedded input set (`norm_cross` on the cross stream).
- `blocks.1, blocks.2` : **self-attn (attn1)** + ff.
- `norm_out` : LayerNorm(768).

Block attn weights (diffusers `Attention`): `to_q/to_k/to_v` (768→768, no bias),
`to_out.0` (768→768 + bias). Cross block carries `attn2.norm_cross` (LayerNorm 768 on KV).

Encoder dataflow (`get_qkv` + `Tripo2Encoder`):
- input = vertices+normals (encoder, 6-ch) / vertices+normals cond (cond, 6-ch), positions
  Fourier-embedded then concatenated with raw features → proj_in → 768.
- 32 learned queries cross-attend the embedded set; then 2 self-attn blocks; norm_out.
Outputs: `quant`/`cond_quant` (below) project 768 → 512.

## Quantizer linears

- `cond_quant` : Linear(768 → 512)  [ne [512,768]] — cond-encoder out → cond latent (512)
- `quant`      : Linear(768 → 512)  — encoder out → z (512) (encode side; FSQ.project_in then 512→5)
- `post_quant` : Linear(512 → 768)  [ne [768,512]] — FSQ code(512) → decoder width (768)

NB the `latent_channels=512` == FSQ `dim`. `indices_to_codes` already returns 512-dim codes
(post FSQ.project_out), which feed `post_quant` after concatenation with the cond latent.

## Decoder (`Tripo2Decoder`, 10 layers → 11 blocks)

151 tensors:
- `proj_query` : Linear(54 → 768)  — per-joint query points (cond-channels + Fourier embed)
- `blocks.0..9` : **self-attn (attn1)** + ff  (10 self-attn blocks)
- `blocks.10`  : **cross-attn (attn2)** + ff  — the query stream cross-attends the latent memory.
- `norm_out` : LayerNorm(768)
- `proj_out` : Linear(768 → 1)  [ne [1,768]] → the per-(joint,point) skin **logit**.

### Decode dataflow (`SkinFSQCVAEModel._decode`, tokenrig.decode loop)
For the skin phase (per `tokenrig.decode`, ~L574, group g = tokens_per_skin·encode_repeat):
1. skin tokens → vae indices (offset, below) → `z = FSQ.indices_to_codes(idx)` → reshape
   `[b, tokens_per_skin=4, 512]`.
2. `_decode(z, cond=cond_latents, sampled_points=sampled_cond)`:
   - `z = post_quant( cat([z, cond], dim=1) )`  → memory tokens at width 768.
     (cond = the cond_encoder's 384-token cond latent for the mesh; z = 4 code tokens)
   - queries = `cat( embed(sampled_points.xyz), sampled_points.features )` → proj_query → 768.
     sampled_points = the N surface points (vertices+normals) the skin is predicted **for**.
   - 10 self-attn blocks over the queries, then block10 cross-attends the memory (z+cond),
     norm_out, `proj_out` → `[b, N, 1]` logits.
3. `skin_pred = sigmoid?` — NOTE: `_decode` returns **raw logits** `[b, N, 1]`. tokenrig.decode
   reshapes to `[N, b]` and concatenates per joint → `skin_pred [N, J]`. The golden `skin_pred.npy`
   is `[8192, 60]`. **Verify whether a sigmoid is applied** (decode returns logits; the saved
   `skin_pred` may be logits or post-sigmoid — check `tokenrig` / SkinVAEModel before the decoder
   port so the final activation matches). The task brief says "logits[N,J] then sigmoid".

## model-vocab → vae-index offset  (CONFIRMED)

`tokenrig.py::decode()` (~L576): `indices = skin_tokens - tokenizer.vocab_size`.
`tokenizer.vocab_size == 267` (== meta.json `tokenizer_vocab`). So:

    vae_index = model_token_id - 267

Token layout (also documented in `rig_grammar.hpp`):
- `[0, 267)` skeleton/tokenizer ids (skin phase starts after the **tokenizer-eos = 258**)
- `[267, 267+32768)` VAE skin codes
- model-eos = 33035 (= vocab_size−1) terminates the skin phase.

The e2e golden emitted **239** valid skin tokens (then model-eos), not a full J·tps=240 — so this
golden's `decode()` shape-check (239≠240) would actually skip skin decode in the strict path; the
saved `skin_pred [8192,60]` came from the model's own run. For FSQ validation we use the 239 valid
vae-range tokens; the port is bit-exact regardless.

## GGUF

- Source weights: `/mnt/hdd/3d/avatar-shootout/_weights/skin_vae_w/` (252 fp32 .npy, names =
  ckpt key minus the leading `vae.` prefix, e.g. `model.FSQ.project_out.weight`).
- Packed (via `pack_gguf`, f32): `/mnt/hdd/3d/avatar-shootout/_weights/skin_vae_gguf/skin_vae.gguf`
  — **487 MB, 252 tensors, all f32 (type 0)**. ne = reversed npy shape (torch [out,in] → ne [in,out]).
  Loads cleanly via `gguf_reader.hpp`. (All tensors ≤2D — no conv, so the >4D collapse path is unused.)

## Open items for the follow-up decoder port

- **Fourier/PMPE exact formula** (`embeddings.py`, `use_pmpe=True`): confirm out_dim=51 and the
  frequency/phase layout; encoder/decoder embeds must match bit-for-bit.
- **Final activation**: confirm sigmoid vs raw-logits on the `_decode` output before comparing to
  `skin_pred.npy`.
- **cond latent provenance**: decoder memory = `cat([z(4 tokens), cond_latents(384 tokens)])` →
  post_quant. cond_latents come from the cond_encoder (golden `cond_latents.npy` exists for replay).
- diffusers `Attention` numerics (scale = 1/sqrt(head_dim=64), GEGLU ff, norm placement) to mirror.
- Reuse `vecset_encoder.hpp` Tripo2 attention primitives (same block shape) for enc/cond/decoder.
