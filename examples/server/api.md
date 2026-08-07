# stable-diffusion.cpp Server APIs

This document describes the server-facing APIs exposed by `examples/server`.

The server currently exposes three API families:

- `OpenAI API` under `/v1/...`
- `Stable Diffusion WebUI API` under `/sdapi/v1/...`
- `sdcpp API` under `/sdcpp/v1/...`

The `sdcpp API` is the native API surface.
Its request schema is the same schema used by `sd_cpp_extra_args`.

Global LoRA rule:

- Server APIs do not parse LoRA tags embedded inside `prompt`.
- `<lora:...>` prompt syntax is intentionally unsupported in `OpenAI API`, `sdapi`, and `sdcpp API`.
- LoRA must be passed through structured API fields when the API supports it.

## Overview

### OpenAI API

Compatibility API shaped like OpenAI image endpoints.

Current generation-related endpoints include:

- `POST /v1/images/generations`
- `POST /v1/images/edits`
- `GET /v1/models`

### Stable Diffusion WebUI API

Compatibility API shaped like the AUTOMATIC1111 / WebUI endpoints.

Current generation-related endpoints include:

- `POST /sdapi/v1/txt2img`
- `POST /sdapi/v1/img2img`
- `GET /sdapi/v1/loras`
- `GET /sdapi/v1/upscalers`
- `GET /sdapi/v1/latent-upscale-modes`
- `GET /sdapi/v1/samplers`
- `GET /sdapi/v1/schedulers`
- `GET /sdapi/v1/sd-models`
- `GET /sdapi/v1/options`

### sdcpp API

Native async API for `stable-diffusion.cpp`.

Current endpoints include:

- `GET /sdcpp/v1/capabilities`
- `POST /sdcpp/v1/img_gen`
- `GET /sdcpp/v1/jobs/{id}`
- `POST /sdcpp/v1/jobs/{id}/cancel`
- `POST /sdcpp/v1/vid_gen`

## `sd_cpp_extra_args`

`sd_cpp_extra_args` is an extension mechanism for the compatibility APIs.

Rules:

- Its JSON schema is the same schema used by the native `sdcpp API`.
- `OpenAI API` and `sdapi` can embed it inside `prompt`.
- `sdcpp API` does not need it, because the request body already uses the native schema directly.

Embedding format:

```text
normal prompt text <sd_cpp_extra_args>{"sample_params":{"sample_steps":28}}</sd_cpp_extra_args>
```

Behavior:

- The server extracts the JSON block.
- The JSON block is parsed using the same field rules as the `sdcpp API`.
- The block is removed from the final prompt before generation.

Supported use:

- extend `OpenAI API` requests with native `stable-diffusion.cpp` controls
- extend `sdapi` requests with native `stable-diffusion.cpp` controls

Unsupported use:

- do not use `sd_cpp_extra_args` with `/sdcpp/v1/*`

## OpenAI API

### Purpose

This family exists for client compatibility.

Use it when you want OpenAI-style request and response shapes.

### Native Extension

`OpenAI API` supports `sd_cpp_extra_args` embedded inside `prompt`.

The embedded JSON follows the `sdcpp API` request schema.

### Supported Fields

#### `POST /v1/images/generations`

Currently supported top-level request fields:

| Field | Type | Notes |
| --- | --- | --- |
| `prompt` | `string` | Required |
| `n` | `integer` | Number of images |
| `size` | `string` | Format `WIDTHxHEIGHT` |
| `output_format` | `string` | `png`, `jpeg`, or `webp` |
| `output_compression` | `integer` | Range is clamped to `0..100` |

Native extension fields:

- any `sdcpp API` fields embedded through `sd_cpp_extra_args` inside `prompt`

Response fields:

| Field | Type | Notes |
| --- | --- | --- |
| `created` | `integer` | Unix timestamp |
| `output_format` | `string` | Final encoded image format |
| `data` | `array<object>` | Generated image list |
| `data[].b64_json` | `string` | Base64-encoded image bytes |

#### `POST /v1/images/edits`

Currently supported multipart form fields:

| Field | Type | Notes |
| --- | --- | --- |
| `prompt` | `string` | Required |
| `image[]` | `file[]` | Preferred image upload field |
| `image` | `file` | Legacy single-image upload field |
| `mask` | `file` | Optional mask image |
| `n` | `integer` | Number of images |
| `size` | `string` | Format `WIDTHxHEIGHT` |
| `output_format` | `string` | `png` or `jpeg` |
| `output_compression` | `integer` | Range is clamped to `0..100` |

Native extension fields:

- any `sdcpp API` fields embedded through `sd_cpp_extra_args` inside `prompt`

Response fields:

| Field | Type | Notes |
| --- | --- | --- |
| `created` | `integer` | Unix timestamp |
| `output_format` | `string` | Final encoded image format |
| `data` | `array<object>` | Generated image list |
| `data[].b64_json` | `string` | Base64-encoded image bytes |

#### `GET /v1/models`

Response fields:

| Field | Type | Notes |
| --- | --- | --- |
| `data` | `array<object>` | Available local models |
| `data[].id` | `string` | Currently fixed to `sd-cpp-local` |
| `data[].object` | `string` | Currently fixed to `model` |
| `data[].owned_by` | `string` | Currently fixed to `local` |

### Output Options

`OpenAI API` supports response serialization controls such as:

- `output_format`
- `output_compression`

### Notes

- `OpenAI API` is synchronous from the HTTP client's perspective.
- Native async job polling is not exposed through this family.
- Prompt-embedded `<lora:...>` tags are intentionally unsupported.

## Stable Diffusion WebUI API

### Purpose

This family exists for client compatibility with WebUI-style tools.

Use it when you want `txt2img` / `img2img`-style endpoints and response shapes.

### Native Extension

`sdapi` supports `sd_cpp_extra_args` embedded inside `prompt`.

The embedded JSON follows the `sdcpp API` request schema.

This allows `sdapi` clients to use native `stable-diffusion.cpp` controls without changing the outer request format.

### Supported Fields

#### `POST /sdapi/v1/txt2img`

Currently supported request fields:

| Field | Type | Notes |
| --- | --- | --- |
| `prompt` | `string` | Required |
| `negative_prompt` | `string` | Optional |
| `width` | `integer` | Positive image width |
| `height` | `integer` | Positive image height |
| `steps` | `integer` | Sampling steps |
| `cfg_scale` | `number` | Text CFG scale |
| `seed` | `integer` | `-1` means random |
| `batch_size` | `integer` | Number of images |
| `clip_skip` | `integer` | Optional |
| `sampler_name` | `string` | WebUI sampler name |
| `scheduler` | `string` | Scheduler name |
| `lora` | `array<object>` | Structured LoRA list |
| `extra_images` | `array<string>` | Base64 or data URL images |
| `enable_hr` | `boolean` | Enable highres fix for `txt2img` |
| `hr_upscaler` | `string` | `Lanczos`, `Nearest`, a latent mode such as `Latent (nearest-exact)`, or an upscaler model name from `/sdapi/v1/upscalers` |
| `hr_scale` | `number` | Highres scale when resize target is not set |
| `hr_resize_x` | `integer` | Highres target width, `0` to use scale |
| `hr_resize_y` | `integer` | Highres target height, `0` to use scale |
| `hr_steps` | `integer` | Highres second-pass sample steps, `0` to reuse `steps` |
| `denoising_strength` | `number` | Highres denoising strength for `txt2img` |

Native extension fields:

- any `sdcpp API` fields embedded through `sd_cpp_extra_args` inside `prompt`

Response fields:

| Field | Type | Notes |
| --- | --- | --- |
| `images` | `array<string>` | Base64-encoded PNG images |
| `parameters` | `object` | Echo of the parsed outer request body |
| `info` | `string` | Currently empty string |

#### `POST /sdapi/v1/img2img`

Currently supported request fields:

| Field | Type | Notes |
| --- | --- | --- |
| all currently supported `txt2img` fields | same as above | Reused |
| `init_images` | `array<string>` | Base64 or data URL images |
| `mask` | `string` | Base64 or data URL image |
| `inpainting_mask_invert` | `integer` or `boolean` | Treated as invert flag |
| `denoising_strength` | `number` | Clamped to `0.0..1.0` |

Highres fix fields are currently handled for `txt2img`; `img2img` uses `denoising_strength` as image-to-image strength.

Native extension fields:

- any `sdcpp API` fields embedded through `sd_cpp_extra_args` inside `prompt`

Response fields:

| Field | Type | Notes |
| --- | --- | --- |
| `images` | `array<string>` | Base64-encoded PNG images |
| `parameters` | `object` | Echo of the parsed outer request body |
| `info` | `string` | Currently empty string |

#### Discovery / Compatibility Endpoints

Currently exposed:

- `GET /sdapi/v1/loras`
- `GET /sdapi/v1/upscalers`
- `GET /sdapi/v1/latent-upscale-modes`
- `GET /sdapi/v1/samplers`
- `GET /sdapi/v1/schedulers`
- `GET /sdapi/v1/sd-models`
- `GET /sdapi/v1/options`

Response fields:

`GET /sdapi/v1/loras`

| Field | Type | Notes |
| --- | --- | --- |
| `[].name` | `string` | Display name derived from file stem |
| `[].path` | `string` | Relative path under the configured LoRA directory |

`GET /sdapi/v1/upscalers`

| Field | Type | Notes |
| --- | --- | --- |
| `[].name` | `string` | Built-in name or model stem |
| `[].model_name` | `string \| null` | Model family label for model-backed upscalers |
| `[].model_path` | `string \| null` | Absolute model path for model-backed upscalers |
| `[].model_url` | `string \| null` | Currently always null |
| `[].scale` | `integer` | Currently `4` |

Built-in entries include `None`, `Lanczos`, and `Nearest`. Model-backed entries are scanned from the top level of `--hires-upscalers-dir`; subdirectories are not scanned.

`GET /sdapi/v1/latent-upscale-modes`

| Field | Type | Notes |
| --- | --- | --- |
| `[].name` | `string` | WebUI-compatible latent upscale mode name |

Built-in latent modes include `Latent`, `Latent (nearest)`, `Latent (nearest-exact)`, `Latent (antialiased)`, `Latent (bicubic)`, and `Latent (bicubic antialiased)`.

`GET /sdapi/v1/samplers`

| Field | Type | Notes |
| --- | --- | --- |
| `[].name` | `string` | Sampler name |
| `[].aliases` | `array<string>` | Currently contains the same single sampler name |
| `[].options` | `object` | Currently empty object |

`GET /sdapi/v1/schedulers`

| Field | Type | Notes |
| --- | --- | --- |
| `[].name` | `string` | Scheduler name |
| `[].label` | `string` | Same value as `name` |

`GET /sdapi/v1/sd-models`

| Field | Type | Notes |
| --- | --- | --- |
| `[].title` | `string` | Model stem |
| `[].model_name` | `string` | Same value as `title` |
| `[].filename` | `string` | Model filename |
| `[].hash` | `string` | Placeholder compatibility value |
| `[].sha256` | `string` | Placeholder compatibility value |
| `[].config` | `null` | Currently always null |

`GET /sdapi/v1/options`

| Field | Type | Notes |
| --- | --- | --- |
| `samples_format` | `string` | Currently fixed to `png` |
| `sd_model_checkpoint` | `string` | Model stem |

### Notes

- `sdapi` is synchronous from the HTTP client's perspective.
- Prompt-embedded `<lora:...>` tags are intentionally unsupported.

## sdcpp API

### Purpose

This is the native `stable-diffusion.cpp` API.

Use it when you want:

- async job submission
- explicit native parameter control
- frontend-oriented capability discovery

### Job Model

All async generation requests create a job.

Job states:

- `queued`
- `generating`
- `completed`
- `failed`
- `cancelled`

Common job shape:

```json
{
  "id": "job_01HTXYZABC",
  "kind": "img_gen",
  "status": "queued",
  "created": 1775401200,
  "started": null,
  "completed": null,
  "queue_position": 2,
  "result": null,
  "error": null
}
```

Field types:

| Field | Type |
| --- | --- |
| `id` | `string` |
| `kind` | `string` |
| `status` | `string` |
| `created` | `integer` |
| `started` | `integer \| null` |
| `completed` | `integer \| null` |
| `queue_position` | `integer` |
| `result` | `object \| null` |
| `error` | `object \| null` |

### Endpoints

#### `GET /sdcpp/v1/capabilities`

Returns frontend-friendly capability metadata.

The mode-aware fields are the primary interface. The top-level compatibility fields are deprecated mirrors kept for older clients.

Top-level fields:

| Field | Type | Notes |
| --- | --- | --- |
| `model` | `object` | Loaded model metadata |
| `current_mode` | `string` | The native generation mode mirrored by top-level compatibility fields |
| `supported_modes` | `array<string>` | Supported native modes such as `img_gen` or `vid_gen` |
| `defaults` | `object` | Deprecated compatibility mirror of `defaults_by_mode[current_mode]` |
| `output_formats` | `array<string>` | Deprecated compatibility mirror of `output_formats_by_mode[current_mode]` |
| `features` | `object` | Deprecated compatibility mirror of `features_by_mode[current_mode]` |
| `defaults_by_mode` | `object` | Explicit defaults for each supported mode |
| `output_formats_by_mode` | `object` | Explicit output formats for each supported mode |
| `features_by_mode` | `object` | Explicit feature flags for each supported mode |
| `samplers` | `array<string>` | Available sampling methods |
| `schedulers` | `array<string>` | Available schedulers |
| `loras` | `array<object>` | Available LoRA entries |
| `upscalers` | `array<object>` | Available model-backed highres upscalers |
| `limits` | `object` | Shared queue and size limits |

`model`

| Field | Type |
| --- | --- |
| `model.name` | `string` |
| `model.stem` | `string` |
| `model.path` | `string` |

Compatibility rules:

- `defaults`, `output_formats`, and `features` are deprecated compatibility mirrors
- those three top-level fields always mirror `current_mode`
- `supported_modes`, `defaults_by_mode`, `output_formats_by_mode`, and `features_by_mode` are the mode-aware fields

Mode-aware objects:

| Field | Type |
| --- | --- |
| `defaults_by_mode.img_gen` | `object` |
| `defaults_by_mode.vid_gen` | `object` |
| `output_formats_by_mode.img_gen` | `array<string>` |
| `output_formats_by_mode.vid_gen` | `array<string>` |
| `features_by_mode.img_gen` | `object` |
| `features_by_mode.vid_gen` | `object` |

Shared nested fields:

`loras`

| Field | Type |
| --- | --- |
| `loras[].name` | `string` |
| `loras[].path` | `string` |

`upscalers`

| Field | Type | Notes |
| --- | --- | --- |
| `upscalers[].name` | `string` | Built-in name or model stem; use this value in `hires.upscaler` |

Built-in entries include `None`, `Lanczos`, `Nearest`, `Latent`, `Latent (nearest)`, `Latent (nearest-exact)`, `Latent (antialiased)`, `Latent (bicubic)`, and `Latent (bicubic antialiased)`. Model-backed entries are scanned from the top level of `--hires-upscalers-dir`; subdirectories are not scanned.

`limits`

| Field | Type |
| --- | --- |
| `limits.min_width` | `integer` |
| `limits.max_width` | `integer` |
| `limits.min_height` | `integer` |
| `limits.max_height` | `integer` |
| `limits.max_batch_count` | `integer` |
| `limits.max_queue_size` | `integer` |

Shared default fields used by both `img_gen` and `vid_gen`:

| Field | Type |
| --- | --- |
| `prompt` | `string` |
| `negative_prompt` | `string` |
| `clip_skip` | `integer` |
| `width` | `integer` |
| `height` | `integer` |
| `strength` | `number` |
| `seed` | `integer` |
| `sample_params` | `object` |
| `sample_params.scheduler` | `string` |
| `sample_params.sample_method` | `string` |
| `sample_params.sample_steps` | `integer` |
| `sample_params.eta` | `number \| null` |
| `sample_params.shifted_timestep` | `integer` |
| `sample_params.flow_shift` | `number \| null` |
| `sample_params.guidance.txt_cfg` | `number` |
| `sample_params.guidance.img_cfg` | `number \| null` |
| `sample_params.guidance.distilled_guidance` | `number` |
| `sample_params.guidance.slg.layers` | `array<integer>` |
| `sample_params.guidance.slg.layer_start` | `number` |
| `sample_params.guidance.slg.layer_end` | `number` |
| `sample_params.guidance.slg.scale` | `number` |
| `vae_tiling_params` | `object` |
| `vae_tiling_params.enabled` | `boolean` |
| `vae_tiling_params.temporal_tiling` | `boolean` |
| `vae_tiling_params.tile_size_x` | `integer` |
| `vae_tiling_params.tile_size_y` | `integer` |
| `vae_tiling_params.target_overlap` | `number` |
| `vae_tiling_params.rel_size_x` | `number` |
| `vae_tiling_params.rel_size_y` | `number` |
| `vae_tiling_params.extra_tiling_args` | `string` |
| `cache_mode` | `string` |
| `cache_option` | `string` |
| `scm_mask` | `string` |
| `scm_policy_dynamic` | `boolean` |
| `output_format` | `string` |
| `output_compression` | `integer` |

`vae_tiling_params.extra_tiling_args` accepts a key=value list. For LTX video VAE temporal tiling, `temporal_tile_frames` defaults to `4` and `temporal_tile_overlap` defaults to `1`.

`img_gen`-specific default fields:

| Field | Type |
| --- | --- |
| `batch_count` | `integer` |
| `auto_resize_ref_image` | `boolean` |
| `increase_ref_index` | `boolean` |
| `ref_image_args` | `string` |
| `control_strength` | `number` |
| `ip_adapter_strength` | `number` |
| `hires` | `object` |
| `hires.enabled` | `boolean` |
| `hires.upscaler` | `string` |
| `hires.scale` | `number` |
| `hires.target_width` | `integer` |
| `hires.target_height` | `integer` |
| `hires.steps` | `integer` |
| `hires.denoising_strength` | `number` |
| `hires.custom_sigmas` | `array<number>` |
| `hires.upscale_tile_size` | `integer` |

`vid_gen`-specific default fields:

| Field | Type |
| --- | --- |
| `video_frames` | `integer` |
| `fps` | `integer` |
| `moe_boundary` | `number` |
| `vace_strength` | `number` |
| `audio_path` | `string \| null` |
| `audio_frame_offset` | `integer` |
| `high_noise_sample_params` | `object` |
| `high_noise_sample_params.scheduler` | `string` |
| `high_noise_sample_params.sample_method` | `string` |
| `high_noise_sample_params.sample_steps` | `integer` |
| `high_noise_sample_params.eta` | `number \| null` |
| `high_noise_sample_params.shifted_timestep` | `integer` |
| `high_noise_sample_params.flow_shift` | `number \| null` |
| `high_noise_sample_params.guidance.txt_cfg` | `number` |
| `high_noise_sample_params.guidance.img_cfg` | `number \| null` |
| `high_noise_sample_params.guidance.distilled_guidance` | `number` |
| `high_noise_sample_params.guidance.slg.layers` | `array<integer>` |
| `high_noise_sample_params.guidance.slg.layer_start` | `number` |
| `high_noise_sample_params.guidance.slg.layer_end` | `number` |
| `high_noise_sample_params.guidance.slg.scale` | `number` |

Fields returned in `features_by_mode.img_gen`:

- `init_image`
- `mask_image`
- `control_image`
- `ip_adapter_image`
- `ref_images`
- `ref_image_args`
- `lora`
- `vae_tiling`
- `hires`
- `cache`
- `cancel_queued`
- `cancel_generating`

Fields returned in `features_by_mode.vid_gen`:

- `init_image`
- `end_image`
- `control_frames`
- `high_noise_sample_params`
- `lora`
- `vae_tiling`
- `cache`
- `cancel_queued`
- `cancel_generating`

#### `POST /sdcpp/v1/img_gen`

Submits an async image generation job.

Successful submission returns `202 Accepted`.

Example response:

```json
{
  "id": "job_01HTXYZABC",
  "kind": "img_gen",
  "status": "queued",
  "created": 1775401200,
  "poll_url": "/sdcpp/v1/jobs/job_01HTXYZABC"
}
```

Response fields:

| Field | Type |
| --- | --- |
| `id` | `string` |
| `kind` | `string` |
| `status` | `string` |
| `created` | `integer` |
| `poll_url` | `string` |

#### `GET /sdcpp/v1/jobs/{id}`

Returns current job status.

Typical status codes:

- `200 OK`
- `404 Not Found`
- `410 Gone`

#### `POST /sdcpp/v1/jobs/{id}/cancel`

Attempts to cancel an accepted job.

Typical status codes:

- `200 OK`
- `404 Not Found`
- `409 Conflict`
- `410 Gone`

### Request Body

Example:

```json
{
  "prompt": "a cat sitting on a chair",
  "negative_prompt": "",
  "clip_skip": -1,
  "width": 1024,
  "height": 1024,
  "strength": 0.75,
  "seed": -1,
  "batch_count": 1,
  "auto_resize_ref_image": true,
  "increase_ref_index": false,
  "control_strength": 0.9,
  "ip_adapter_strength": 1.0,
  "embed_image_metadata": true,

  "ref_image_args": "",

  "init_image": null,
  "ref_images": [],
  "mask_image": null,
  "control_image": null,
  "ip_adapter_image": null,

  "sample_params": {
    "scheduler": "discrete",
    "sample_method": "euler_a",
    "sample_steps": 28,
    "eta": 1.0,
    "shifted_timestep": 0,
    "custom_sigmas": [],
    "flow_shift": 0.0,
    "guidance": {
      "txt_cfg": 7.0,
      "img_cfg": 7.0,
      "distilled_guidance": 3.5,
      "slg": {
        "layers": [7, 8, 9],
        "layer_start": 0.01,
        "layer_end": 0.2,
        "scale": 0.0
      }
    }
  },

  "lora": [],
  "hires": {
    "enabled": false,
    "upscaler": "Latent",
    "scale": 2.0,
    "target_width": 0,
    "target_height": 0,
    "steps": 0,
    "denoising_strength": 0.7,
    "custom_sigmas": [],
    "upscale_tile_size": 128
  },

  "vae_tiling_params": {
    "enabled": false,
    "temporal_tiling": false,
    "tile_size_x": 0,
    "tile_size_y": 0,
    "target_overlap": 0.5,
    "rel_size_x": 0.0,
    "rel_size_y": 0.0,
    "extra_tiling_args": ""
  },

  "cache_mode": "disabled",
  "cache_option": "",
  "scm_mask": "",
  "scm_policy_dynamic": true,

  "output_format": "png",
  "output_compression": 100
}
```

### LoRA Rules

- The server only accepts explicit LoRA entries from the `lora` field.
- Prompt-embedded `<lora:...>` tags are intentionally unsupported.
- Clients should resolve LoRA usage through the structured `lora` array.

### Image Encoding Rules

Any image field accepts:

- a raw base64 string, or
- a data URL such as `data:image/png;base64,...`, or
- `part:<name>`, naming a binary part of the SAME multipart request.

`part:` is the preferred form and the reason `/sdcpp/v1/img_gen` and `/sdcpp/v1/vid_gen` accept a
multipart body at all: send a `request` part holding the JSON document plus one binary part per
image, and no artifact ever becomes a JSON string. A 194 KiB control frame costs 259 KiB inlined,
and a relip window carries 65 of them. Base64 stays accepted indefinitely — the two forms may be
mixed in one request, per field.

Part names are free-form with one convention: a name shaped `a_<16 lowercase hex>` is read as the
first 16 hex digits of the part's SHA-256, and the server VERIFIES it on receipt. A mismatch is a
400 naming the part, rather than a subtly wrong render returning 200. Any other name is taken as
given and not checked.

An unknown part name fails exactly as undecodable base64 does: a 400 naming the field that held it.

Servers that understand `part:` say so in `GET /health` as `"accepts_part_refs": true`. Gate on
that flag rather than on deploy order — an older engine reads the marker as base64 and fails.

Channel expectations:

- `init_image`: 3 channels
- `ref_images[]`: 3 channels
- `control_image`: 3 channels
- `ip_adapter_image`: 3 channels
- `mask_image`: 1 channel

If omitted or null:

- single-image fields map to an empty `sd_image_t`
- array fields map to an empty C-style array, represented as `pointer = nullptr` and `count = 0`

### Field Mapping Summary

Top-level scalar fields:

| Field | Type |
| --- | --- |
| `prompt` | `string` |
| `negative_prompt` | `string` |
| `clip_skip` | `integer` |
| `width` | `integer` |
| `height` | `integer` |
| `strength` | `number` |
| `seed` | `integer` |
| `batch_count` | `integer` |
| `auto_resize_ref_image` | `boolean` |
| `increase_ref_index` | `boolean` |
| `ref_image_args` | `string` |
| `control_strength` | `number` |
| `ip_adapter_strength` | `number` |
| `embed_image_metadata` | `boolean` |

Image fields:

| Field | Type |
| --- | --- |
| `init_image` | `string \| null` |
| `ref_images` | `array<string>` |
| `mask_image` | `string \| null` |
| `control_image` | `string \| null` |
| `ip_adapter_image` | `string \| null` |

`ref_image_args` is a comma-separated `key=value` list controlling how `ref_images`
are prepared, overriding the server's `--ref-image-args` default for this job.
Start with `preset=<name>`; later keys override the preset. Presets:
`flux_kontext`, `flux2`, `longcat`, `qwen`, `qwen_layered`, `mage_flow`,
`z_image_omni`, `cosmos_reference`, `krea2_ostris_edit`, `krea2_edit`,
`krea2_identity_edit`. Keys: `pass_to_vlm`, `pass_to_dit`, `ref_index_mode`
(`fixed`/`increase`/`decrease`), `force_ref_timestep_zero`, `resize_before_vae`,
`resize_vae_to_target`, `crop_vae_to_target_ar`, `vae_input_max_pixels`,
`vlm_resize_mode` (`area`/`longest_side`/`none`), `vlm_min_size`, `vlm_max_size`,
`vlm_size`, `vlm_picture_labels`.

Krea 2 edit example (conradlocke/krea2-identity-edit + its LoRA):

```json
{
  "prompt": "put this person in a night market, same face",
  "width": 1024, "height": 1024, "steps": 10, "cfg_scale": 1.0,
  "ref_images": ["<base64 png>"],
  "ref_image_args": "preset=krea2_identity_edit",
  "lora": [{"path": "krea2-identity-edit-v1_2", "multiplier": 1.0}]
}
```

LoRA fields:

| Field | Type |
| --- | --- |
| `lora[].path` | `string` |
| `lora[].multiplier` | `number` |
| `lora[].is_high_noise` | `boolean` |

Sampling fields:

| Field | Type |
| --- | --- |
| `sample_params.scheduler` | `string` |
| `sample_params.sample_method` | `string` |
| `sample_params.sample_steps` | `integer` |
| `sample_params.eta` | `number` |
| `sample_params.shifted_timestep` | `integer` |
| `sample_params.custom_sigmas` | `array<number>` |
| `sample_params.flow_shift` | `number` |
| `sample_params.guidance.txt_cfg` | `number` |
| `sample_params.guidance.img_cfg` | `number` |
| `sample_params.guidance.distilled_guidance` | `number` |
| `sample_params.guidance.slg.layers` | `array<integer>` |
| `sample_params.guidance.slg.layer_start` | `number` |
| `sample_params.guidance.slg.layer_end` | `number` |
| `sample_params.guidance.slg.scale` | `number` |

Other native fields:

| Field | Type |
| --- | --- |
| `hires` | `object` |
| `hires.enabled` | `boolean` |
| `hires.upscaler` | `string` |
| `hires.scale` | `number` |
| `hires.target_width` | `integer` |
| `hires.target_height` | `integer` |
| `hires.steps` | `integer` |
| `hires.denoising_strength` | `number` |
| `hires.custom_sigmas` | `array<number>` |
| `hires.upscale_tile_size` | `integer` |
| `vae_tiling_params` | `object` |
| `vae_tiling_params.enabled` | `boolean` |
| `vae_tiling_params.temporal_tiling` | `boolean` |
| `vae_tiling_params.tile_size_x` | `integer` |
| `vae_tiling_params.tile_size_y` | `integer` |
| `vae_tiling_params.target_overlap` | `number` |
| `vae_tiling_params.rel_size_x` | `number` |
| `vae_tiling_params.rel_size_y` | `number` |
| `vae_tiling_params.extra_tiling_args` | `string` |
| `cache_mode` | `string` |
| `cache_option` | `string` |
| `scm_mask` | `string` |
| `scm_policy_dynamic` | `boolean` |

For `hires.upscaler`, use `Lanczos`, `Nearest`, `Latent`, `Latent (nearest)`, `Latent (nearest-exact)`, `Latent (antialiased)`, `Latent (bicubic)`, `Latent (bicubic antialiased)`, or an `upscalers[].name` value from `GET /sdcpp/v1/capabilities`. Model-backed upscalers are resolved as `--hires-upscalers-dir / (name + ext)` and must live directly in that directory. `hires.custom_sigmas`, when present, overrides the generated second-pass hires sigma schedule; otherwise the hires schedule is trimmed by `hires.denoising_strength`.

HTTP-only output fields:

| Field | Type |
| --- | --- |
| `output_format` | `string` |
| `output_compression` | `integer` |

### Optional Field Handling

Optional sampling fields may be omitted.

When omitted, backend defaults apply to these fields:

- `sample_params.scheduler`
- `sample_params.sample_method`
- `sample_params.eta`
- `sample_params.flow_shift`
- `sample_params.guidance.img_cfg`

### Completion Result

Example completed job:

```json
{
  "id": "job_01HTXYZABC",
  "kind": "img_gen",
  "status": "completed",
  "created": 1775401200,
  "started": 1775401203,
  "completed": 1775401215,
  "queue_position": 0,
  "result": {
    "output_format": "png",
    "images": [
      {
        "index": 0,
        "bytes": 1483920,
        "b64_json": "iVBORw0KGgoAAA..."
      }
    ]
  },
  "error": null
}
```

Every image is also served RAW at `GET /sdcpp/v1/jobs/{id}/media?index=<index>` — same buffer, no
encode, no decode, a third fewer bytes. `index` defaults to `0`.

`b64_json` is present only while the artifact is inlined; with `SDCPP_JOB_MEDIA_B64=0` in the
server's environment the field is OMITTED (not empty) and `…/media?index=n` is the way to collect
a result. `bytes` is always present, so a client can tell "not inlined" from "empty render" without
fetching anything.

### Failure Result

Example failed job:

```json
{
  "id": "job_01HTXYZABC",
  "kind": "img_gen",
  "status": "failed",
  "created": 1775401200,
  "started": 1775401203,
  "completed": 1775401204,
  "queue_position": 0,
  "result": null,
  "error": {
    "code": "generation_failed",
    "message": "generate_image returned empty results"
  }
}
```

### Cancelled Result

Example cancelled job:

```json
{
  "id": "job_01HTXYZABC",
  "kind": "img_gen",
  "status": "cancelled",
  "created": 1775401200,
  "started": null,
  "completed": 1775401202,
  "queue_position": 0,
  "result": null,
  "error": {
    "code": "cancelled",
    "message": "job cancelled by client"
  }
}
```

### Submission Errors

`POST /sdcpp/v1/img_gen` may return:

- `202 Accepted` when the job is created
- `400 Bad Request` for an empty body, unsupported model mode, invalid JSON, or invalid generation parameters
- `429 Too Many Requests` when the job queue is full
- `500 Internal Server Error` for unexpected server exceptions during submission

### `vid_gen`

The following section documents the native async contract for video generation.

#### `POST /sdcpp/v1/vid_gen`

Submits an async video generation job.

Successful submission returns `202 Accepted`.

Example response:

```json
{
  "id": "job_01HTXYZVID",
  "kind": "vid_gen",
  "status": "queued",
  "created": 1775401200,
  "poll_url": "/sdcpp/v1/jobs/job_01HTXYZVID"
}
```

Response fields:

| Field | Type |
| --- | --- |
| `id` | `string` |
| `kind` | `string` |
| `status` | `string` |
| `created` | `integer` |
| `poll_url` | `string` |

### Request Body

Compared with `img_gen`, the `vid_gen` request body:

- `vid_gen` is a single video sequence job, so `batch_count` is not part of the request schema
- `ref_images`, `mask_image`, `control_image`, `control_strength`, `ip_adapter_image`, `ip_adapter_strength`, and `embed_image_metadata` are not part of the request schema
- `vid_gen` adds `end_image`, `control_frames`, `high_noise_sample_params`, `video_frames`, `fps`, `moe_boundary`, `vace_strength`, and LongCat Avatar's `audio_path` / `audio_frame_offset`

Example:

```json
{
  "prompt": "a cat walking through a rainy alley",
  "negative_prompt": "",
  "clip_skip": -1,
  "width": 832,
  "height": 480,
  "strength": 0.75,
  "seed": -1,
  "video_frames": 33,
  "fps": 16,
  "moe_boundary": 0.875,
  "vace_strength": 1.0,
  "audio_path": null,
  "audio_frame_offset": 0,

  "init_image": null,
  "end_image": null,
  "control_frames": [],

  "sample_params": {
    "scheduler": "discrete",
    "sample_method": "euler",
    "sample_steps": 28,
    "eta": 1.0,
    "shifted_timestep": 0,
    "custom_sigmas": [],
    "flow_shift": 0.0,
    "guidance": {
      "txt_cfg": 7.0,
      "img_cfg": 7.0,
      "distilled_guidance": 3.5,
      "slg": {
        "layers": [7, 8, 9],
        "layer_start": 0.01,
        "layer_end": 0.2,
        "scale": 0.0
      }
    }
  },

  "high_noise_sample_params": {
    "scheduler": "discrete",
    "sample_method": "euler",
    "sample_steps": -1,
    "eta": 1.0,
    "shifted_timestep": 0,
    "flow_shift": 0.0,
    "guidance": {
      "txt_cfg": 7.0,
      "img_cfg": 7.0,
      "distilled_guidance": 3.5,
      "slg": {
        "layers": [7, 8, 9],
        "layer_start": 0.01,
        "layer_end": 0.2,
        "scale": 0.0
      }
    }
  },

  "lora": [],

  "vae_tiling_params": {
    "enabled": false,
    "temporal_tiling": false,
    "tile_size_x": 0,
    "tile_size_y": 0,
    "target_overlap": 0.5,
    "rel_size_x": 0.0,
    "rel_size_y": 0.0,
    "extra_tiling_args": ""
  },

  "cache_mode": "disabled",
  "cache_option": "",
  "scm_mask": "",
  "scm_policy_dynamic": true,

  "output_format": "webm",
  "output_compression": 100
}
```

### LoRA Rules

- The server only accepts explicit LoRA entries from the `lora` field.
- Prompt-embedded `<lora:...>` tags are intentionally unsupported.
- `lora[].is_high_noise` controls whether a LoRA applies only to the high-noise stage.

### Image and Frame Encoding Rules

Any image field accepts:

- a raw base64 string, or
- a data URL such as `data:image/png;base64,...`, or
- `part:<name>`, naming a binary part of the SAME multipart request.

`part:` is the preferred form and the reason `/sdcpp/v1/img_gen` and `/sdcpp/v1/vid_gen` accept a
multipart body at all: send a `request` part holding the JSON document plus one binary part per
image, and no artifact ever becomes a JSON string. A 194 KiB control frame costs 259 KiB inlined,
and a relip window carries 65 of them. Base64 stays accepted indefinitely — the two forms may be
mixed in one request, per field.

Part names are free-form with one convention: a name shaped `a_<16 lowercase hex>` is read as the
first 16 hex digits of the part's SHA-256, and the server VERIFIES it on receipt. A mismatch is a
400 naming the part, rather than a subtly wrong render returning 200. Any other name is taken as
given and not checked.

An unknown part name fails exactly as undecodable base64 does: a 400 naming the field that held it.

Servers that understand `part:` say so in `GET /health` as `"accepts_part_refs": true`. Gate on
that flag rather than on deploy order — an older engine reads the marker as base64 and fails.

Channel expectations:

- `init_image`: 3 channels
- `end_image`: 3 channels
- `control_frames[]`: 3 channels

Frame ordering rules:

- `control_frames[]` order is the conditioning frame order
- `control_frames[]` is preserved in request order

If omitted or null:

- single-image fields map to an empty `sd_image_t`
- array fields map to an empty C-style array, represented as `pointer = nullptr` and `count = 0`

### Field Mapping Summary

Top-level scalar fields:

| Field | Type |
| --- | --- |
| `prompt` | `string` |
| `negative_prompt` | `string` |
| `clip_skip` | `integer` |
| `width` | `integer` |
| `height` | `integer` |
| `strength` | `number` |
| `seed` | `integer` |
| `video_frames` | `integer` |
| `fps` | `integer` |
| `moe_boundary` | `number` |
| `vace_strength` | `number` |
| `audio_path` | `string \| null` |
| `audio_frame_offset` | `integer` |

Image and frame fields:

| Field | Type |
| --- | --- |
| `init_image` | `string \| null` |
| `end_image` | `string \| null` |
| `control_frames` | `array<string>` |

LoRA fields:

| Field | Type |
| --- | --- |
| `lora[].path` | `string` |
| `lora[].multiplier` | `number` |
| `lora[].is_high_noise` | `boolean` |

Sampling fields:

| Field | Type |
| --- | --- |
| `sample_params.scheduler` | `string` |
| `sample_params.sample_method` | `string` |
| `sample_params.sample_steps` | `integer` |
| `sample_params.eta` | `number` |
| `sample_params.shifted_timestep` | `integer` |
| `sample_params.custom_sigmas` | `array<number>` |
| `sample_params.flow_shift` | `number` |
| `sample_params.guidance.txt_cfg` | `number` |
| `sample_params.guidance.img_cfg` | `number` |
| `sample_params.guidance.distilled_guidance` | `number` |
| `sample_params.guidance.slg.layers` | `array<integer>` |
| `sample_params.guidance.slg.layer_start` | `number` |
| `sample_params.guidance.slg.layer_end` | `number` |
| `sample_params.guidance.slg.scale` | `number` |

High-noise sampling fields:

| Field | Type |
| --- | --- |
| `high_noise_sample_params.scheduler` | `string` |
| `high_noise_sample_params.sample_method` | `string` |
| `high_noise_sample_params.sample_steps` | `integer` |
| `high_noise_sample_params.eta` | `number` |
| `high_noise_sample_params.shifted_timestep` | `integer` |
| `high_noise_sample_params.flow_shift` | `number` |
| `high_noise_sample_params.guidance.txt_cfg` | `number` |
| `high_noise_sample_params.guidance.img_cfg` | `number` |
| `high_noise_sample_params.guidance.distilled_guidance` | `number` |
| `high_noise_sample_params.guidance.slg.layers` | `array<integer>` |
| `high_noise_sample_params.guidance.slg.layer_start` | `number` |
| `high_noise_sample_params.guidance.slg.layer_end` | `number` |
| `high_noise_sample_params.guidance.slg.scale` | `number` |

Other native fields:

| Field | Type |
| --- | --- |
| `vae_tiling_params` | `object` |
| `vae_tiling_params.enabled` | `boolean` |
| `vae_tiling_params.temporal_tiling` | `boolean` |
| `vae_tiling_params.tile_size_x` | `integer` |
| `vae_tiling_params.tile_size_y` | `integer` |
| `vae_tiling_params.target_overlap` | `number` |
| `vae_tiling_params.rel_size_x` | `number` |
| `vae_tiling_params.rel_size_y` | `number` |
| `vae_tiling_params.extra_tiling_args` | `string` |
| `cache_mode` | `string` |
| `cache_option` | `string` |
| `scm_mask` | `string` |
| `scm_policy_dynamic` | `boolean` |

HTTP-only output fields:

| Field | Type |
| --- | --- |
| `output_format` | `string` |
| `output_compression` | `integer` |

For `vid_gen`, `output_format` and `output_compression` control container encoding.
`fps` is request metadata for the generated sequence and is echoed in the completed job result.

Allowed `output_format` values:

- `webm`
- `webp`
- `avi`

Output format behavior:

- `output_format` defaults to `webm`
- `webp` means animated WebP
- `avi` means MJPG AVI
- `webm` requires the server to be built with WebM support; otherwise the request returns `400`

### Result Payload

Completed jobs return one encoded container payload, not a list of per-frame images.

Result fields:

- `result.b64_json` contains the whole encoded container file as base64 — present ONLY while the
  artifact is inlined. `SDCPP_JOB_MEDIA_B64=0` in the server's environment omits it, and the
  client collects the render from `GET /sdcpp/v1/jobs/{id}/media`, which serves the identical
  buffer raw. A 220 MB render is a 293 MB JSON string inlined, constructed here and parsed there.
- `result.bytes` is the size of the container in bytes, present either way
- `result.mime_type` identifies the media type
- `result.output_format` echoes the selected container format
- `result.fps` echoes the effective playback FPS
- `result.frame_count` reports the actual decoded frame count used to build the container

Expected MIME types:

| `output_format` | `mime_type` |
| --- | --- |
| `webm` | `video/webm` |
| `webp` | `image/webp` |
| `avi` | `video/x-msvideo` |

### Optional Field Handling

Optional sampling fields may be omitted.

When omitted, backend defaults apply to these fields:

- `sample_params.scheduler`
- `sample_params.sample_method`
- `sample_params.eta`
- `sample_params.flow_shift`
- `sample_params.guidance.img_cfg`
- `high_noise_sample_params.scheduler`
- `high_noise_sample_params.sample_method`
- `high_noise_sample_params.eta`
- `high_noise_sample_params.flow_shift`
- `high_noise_sample_params.guidance.img_cfg`

`high_noise_sample_params` may also be omitted entirely.

### Frame Count Semantics

`video_frames` is the requested target length, but the current core video path internally normalizes the effective frame count to the largest `4n + 1` value that does not exceed the requested count.

Examples:

- `video_frames = 33` stays `33`
- `video_frames = 34` becomes `33`
- `video_frames = 32` becomes `29`

The completed job payload includes the actual decoded `frame_count`.

### LTX multi-segment generation

`POST /ltx/v1/generate` accepts the normal video-generation fields plus a non-empty
`segments` array (objects with a `prompt` field) or `prompts` string array. A segment
object may supply `frames` (a positive `8k+1` LTX window length), `scene_cut: true`
for a prompt-only fresh shot, or `init_image` for a fresh image-pinned shot. Fresh shots
are stitched without trimming the preceding continuation overlap. It returns the same
asynchronous job and media URLs as `/sdcpp/v1/vid_gen`.

**`init_image` on segment 0 is not a thing.** Segment 0 is always a fresh scene, so its
opener is the request's **top-level** `init_image`; the per-segment field is only read for
segments `>= 1`. Sending `segments[0].init_image` with no top-level `init_image` folds it up
to the opener (logged once); sending both is a `400`, because there is no defined precedence
between them.

#### Per-shot image-pin strength

A segment object may supply `pin_strength` (`0..1`) to say how hard **that shot** holds its
pinned images — its `init_image`, its `keyframes`, an `end_image`. `1.0` pins exactly (the
conditioned latent frames are held at denoise mask `0`); lower lets the pin flex toward the
generated motion. `0` is legal and means the pin is fully re-denoised.

Omitted, the shot inherits the request's top-level `strength`, which is also what every shot
gets when no segment mentions `pin_strength` at all — such a request takes the identical code
path it did before this field existed, so `pin_strength` absent and `pin_strength` present at
the chain default are the same render, not merely similar ones.

A V2V shot ignores `pin_strength`: on a V2V window `strength` is the SDEdit denoising
schedule, and `v2v_guide_strength` already owns it.

```json
"segments": [
  { "prompt": "she turns to the window", "pin_strength": 1.0 },
  { "prompt": "the room breathes",       "init_image": "<base64>", "pin_strength": 0.6 }
]
```

An object may request `v2v_mode: 1` with `control_frames`, a base64 image array
containing exactly one frame for every requested output frame. This is SDEdit V2V: the
source is VAE-encoded, treated as a fresh shot, and denoised according to optional
`v2v_guide_strength` (`0..1`; otherwise normal `strength`). `v2v_mode: 2` is the
same edit mechanism with an optional trusted `v2v_source_latent_path`: it must name
an existing `seg_<n>.bin` beneath an LTX bank root, avoiding a pixel decode/re-encode
for a prior LTX render. Prefer the portable form `v2v_source_job_id` plus
`v2v_source_segment`; the server resolves that opaque reference in its transient or
persistent bank roots, so Koblem never sends a host filesystem path. Mode 2 accepts
one of those latent forms or `control_frames`, never both; other paths and modes are
rejected.

The server keeps sampled video-only LTX latents in `$LTX_JOB_DIR/<resume_job_id>`
(default `/var/lib/ltx-video/jobs`) and uses them to continue a later request without
re-rendering its completed prefix. Set `resume_job_id` to the prior response's stable
`resume_job_id`; `cont_latent_frames` selects the carried latent overlap (default `3`).
The request's segment list must include the complete prefix plus new tail because the
server reconstructs the stitched media from the durable banks.

#### Character references (TASS overlap conditioning)

An LTX request may carry a top-level `character_refs` array to hold one identity
across shots without spending an i2v guide frame. It pairs with a character-sheet
checkpoint (`model: "faceid-sheet"`) and a prompt prefixed `ref_t2v:`.

```json
"character_refs": [
  { "image": "<base64 png/jpg or absolute path>",
    "source_id": 2,
    "resize_mode": "native_resolution" }
],
"tass_phase_scale": 1.0
```

Each reference is VAE-encoded and appended on the DiT **token** axis with its own
rotary source tag, so it keeps its own spatial grid: a 1536x1024 sheet conditions a
768x448 render at full detail. `resize_mode` defaults to `native_resolution`;
`match_target` resizes to the render bucket first (use it for tight close-up
references, not for sheets). `source_id` is optional and must be `>= 2` and distinct
— zero is the target's own tag and one is reserved; omitted, the engine assigns
`2, 3, 4, ...` in array order, and the same subject should keep the same id across
shots. `tass_phase_scale` defaults to `1.0`, the trained value.

The references apply to every segment of the request, and multiple references in one
shot are architecturally supported but were not trained. All references in one request
must decode to the same resolution. They condition the base pass only, not the
hires-chain refine stages.

References compose with the `LTX_BASE_TEMPORAL_WINDOW` tiling path. The reference
latent is a tensor of its own rather than part of the target grid, so a temporal tile
re-appends the identical reference block to its own frame range and rebuilds only the
per-token tail (positions, source ids, timesteps). Every tile gets the references, not
just the first: a later tile that lost them would be holding the identity through the
frozen overlap alone — the drift the feature exists to remove — and would silently
switch between a tagged and an untagged graph mid-shot. Each tile places the references
on *its own* first frame, so the reference keeps the zero temporal offset it was trained
with; set `LTX_TASS_WINDOW_REF_ABS=1` to pin them at global frame 0 instead. Tiling
still requires the rest of its preconditions — no i2v/keyframe/continuation guide, no
V2V, and either fixed (driving) audio or an audio-free model, since jointly generated
audio cannot be tiled across video windows.

#### `reference_head_trim` — dropping the reference out of frame 0

A TASS reference sits at the target's **latent frame 0** RoPE address, and on some
checkpoints the decoder hands that address straight back: the opening pixel frame of a
bare **t2v** shot renders the reference verbatim. With a location plate in the reference
set, frame 0 is the empty plate (which reads as "the subject never appears"); with only a
character sheet, frame 0 is that sheet's own source photo, background and all.

`reference_head_trim` is an opt-in, request-level (and per-segment) integer that drops the
contaminated head frames from a shot's decoded output:

| value | meaning |
| --- | --- |
| `0` (default) | **off** — byte-identical to a request that never sent the field |
| `-1` | **auto** — the engine derives the amount itself |
| `1..512` | trim exactly that many pixel frames |

```json
{
  "reference_head_trim": -1,
  "segments": [
    { "prompt": "ref_t2v: ...", "reference_head_trim": -1 },
    { "prompt": "...", "reference_head_trim": 0 }
  ]
}
```

A `reference_head_trim` on a segment object overrides the request-level value for that
shot, including an explicit `0` ("off here"). A segment that omits the key inherits.

**Never compute the amount yourself.** Auto is `1 + 8*(K-1)` pixel frames, where `K` is the
largest reference's *latent* frame count — for stills (`K == 1`) exactly **one** pixel
frame — and the engine reads `K` off the references it actually encoded for that shot.
Reference *count* does not change it: every reference is given the same latent-frame
origin, and latent frame 0 is the only latent frame that decodes to a single pixel frame.

The trim **self-gates to a no-op** on any shot that already pins frame 0 — an i2v
`init_image`, a keyframe at index `0`, or a continuation segment — and on any shot that
carries no references in scope. i2v and continuation shots were measured not to leak. Each
gate logs a line naming the reason, so `auto` is safe to leave on for a whole project: only
a bare t2v shot that actually carries references is affected.

The shot is **trimmed, not re-rendered**: it returns `N - trim` frames. Rendering the
frames back is not offered, because LTX requires `frames % 8 == 1` and the smallest legal
increase is a whole latent frame (8 pixel frames). A matching `trim / fps` seconds is cut
off the **head of the output audio** so A/V stays frame-exact; the pre-render drive-audio
window is untouched. The durable bank keeps the shot as rendered — `seg_<n>.bin` is the
full latent and `seg_<n>.len` records the kept length — so a resume or retake reproduces
the same cut.

Whether you want this is a property of the **checkpoint**, which is why it is not
automatic: `echo-e50` leaks and wants `auto`; `msr-v2` and `echo-full` do not leak and
should leave it off.

Set `persist:true` on a new LTX request to store its bank under `$LTX_PERSIST_DIR`
(default `/var/lib/ltx-video/persist`) instead of the FIFO `$LTX_JOB_DIR`; resumes
and V2V job references search both roots. `DELETE /ltx/v1/job?id=<engine-job-id>`
removes an inactive bank (including the requested resume alias) and is idempotent.

### Worker-isolation lifecycle

When `SD_SERVER_WORKER_ISOLATION` (or a service-specific isolation flag) is enabled,
the public server is CUDA-free and proxies generation to one child process. `GET
/health` and `GET /v1/gpu/status` include `loaded` and `worker_pid`; the PID is
`null` while unloaded. `POST /v1/admin/drain` rejects new generation requests, and
`POST /v1/admin/unload` kills and waits for the child before replying with
`cuda_context_released:true` and `worker_pid:null`. If that exit cannot be proved,
unload returns `503` with `cuda_context_released:false`; callers must not schedule
a competing GPU workload. `POST /v1/admin/load` reopens admission; the next
generation lazily starts a new child. A child is also killed when its supervisor
dies, so stopping the public service cannot leave a CUDA worker orphaned. The
filesystem-only `DELETE /ltx/v1/job` cleanup call is handled by the supervisor
without spawning a child, so Director-bank cleanup remains safe after unload.

### LongCat Avatar compatibility endpoint

`POST /generate` preserves the deployed LongCat Avatar wire contract: JSON fields
`image` and `audio` contain base64 image and WAV bytes, and the successful response
is a `video/webm` body. It accepts the normal sampling fields, `segment_frames` as
an alias for `video_frames`, and an optional `bsa` object (`enable`, `radius`,
`self_frame`, `bookend`, `cube_h`, `cube_w`). The route is synchronous and rejects
concurrent renders with `429`; `GET /health` reports its busy/draining/load state.

Set `segments` or `duration_sec` to use latent-tail continuation. `cont_cond_frames`
selects the carried latent tail (default `13`); the route preserves the original WAV
as the delivered audio track across the stitched output.

### Completion Result

Example completed job:

```json
{
  "id": "job_01HTXYZVID",
  "kind": "vid_gen",
  "status": "completed",
  "created": 1775401200,
  "started": 1775401203,
  "completed": 1775401215,
  "queue_position": 0,
  "result": {
    "output_format": "webm",
    "mime_type": "video/webm",
    "fps": 16,
    "frame_count": 33,
    "b64_json": "GkXfo59ChoEBQveBAULygQRC84EIQo..."
  },
  "error": null
}
```

The response returns the encoded `.webm`, animated `.webp`, or `.avi` container payload directly.

### Failure Result

Example failed job:

```json
{
  "id": "job_01HTXYZVID",
  "kind": "vid_gen",
  "status": "failed",
  "created": 1775401200,
  "started": 1775401203,
  "completed": 1775401204,
  "queue_position": 0,
  "result": null,
  "error": {
    "code": "generation_failed",
    "message": "generate_video returned no results"
  }
}
```

### Cancelled Result

Example cancelled job:

```json
{
  "id": "job_01HTXYZVID",
  "kind": "vid_gen",
  "status": "cancelled",
  "created": 1775401200,
  "started": null,
  "completed": 1775401202,
  "queue_position": 0,
  "result": null,
  "error": {
    "code": "cancelled",
    "message": "job cancelled by client"
  }
}
```

### Submission Errors

`POST /sdcpp/v1/vid_gen` may return:

- `202 Accepted` when the job is created
- `400 Bad Request` for an empty body, unsupported model mode, invalid JSON, invalid generation parameters, or an unsupported output format
- `429 Too Many Requests` when the job queue is full
- `500 Internal Server Error` for unexpected server exceptions during submission
