# llama.cpp — Bonsai + TurboQuant fork

> [!IMPORTANT]
> **This is `bonsai-turboquant-llama.cpp`, an experimental integration fork** that brings together two llama.cpp forks into one build:
>
> - **[PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp)** (`prism` branch) — the engine behind the **[Bonsai](https://huggingface.co/collections/prism-ml/bonsai)** ternary models.
> - **[AtomicBot-ai/atomic-llama-cpp-turboquant](https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant)** — the **TurboQuant** low-bit KV-cache / weight quantization fork.
>
> The base history is Prism's `prism` (which itself tracks mainline llama.cpp); TurboQuant's features are ported on top.
> **Goal:** run Bonsai models *and* use TurboQuant from a single build.
>
> **Implemented on this branch (CPU + Vulkan + CUDA):**
>
> - Bonsai ternary weights — `PQ2_0` (ggml id 142), `PTQ1_0` (id 143), plus upstream `Q2_0` (42) / `Q1_0` (41), and the `prism.hadamard.*` weight-fold graph.
> - TurboQuant KV cache — `turbo2` / `turbo3` / `turbo4` (ggml ids 43 / 44 / 47, runtime-only):
>   `llama-cli -m model.gguf -fa on -ctk turbo3 -ctv turbo3`.
> - TurboQuant weights — `TQ3_1S` (45), `TQ4_1S` (46); ConvRot weights `Q8_CR` (48), `Q5_CR` (49), `Q6_CR` (50).
> - `GGML_OP_TURBO_WHT` (Walsh–Hadamard rotation) and the TurboQuant codec (`ggml-turbo-quant.c`).
>
> **Backend coverage:** CPU and Vulkan are implemented and tested. CUDA first-pass support builds with CUDA 13.2 / MSVC 19.51 and passes focused backend tests on an RTX 4060 Laptop GPU, plus a short Bonsai PTQ1_0 + TurboQuant KV generation smoke test. Metal / SYCL TurboQuant kernels are not ported.
>
> On high-GQA models, the default auto-asymmetric policy may upgrade a requested Turbo K cache to `q8_0`. Inspect the startup log for the actual K/V types. `TURBO_AUTO_ASYMMETRIC=0` disables this policy; it was used only for the symmetric Turbo KV smoke tests, not as a quality recommendation.
>
> **CUDA native low-bit attention (default):** TurboQuant KV is dequantized straight into the MMA shared-memory tiles, so the tensor-core FlashAttention kernel reads the low-bit cache in place. No FP16 copy of K/V is written for decode. Supported: NVIDIA, head size 128 or 256, query count up to 8, and K/V pairs `q8_0` + `turbo2/3/4`, `turbo2/turbo2`, `turbo3/turbo3`, `turbo4/turbo4`. Prefill still converts to FP16. `GGML_CUDA_TURBO_FA=0` forces the old conversion path; `=1` also enables the native vector fallback for shapes the MMA path does not cover.
>
> Measured on an RTX 4060 Laptop with the 27B Bonsai PTQ1_0 model and `q8_0` K + `turbo3` V:
>
> | context | FP16 conversion | native low-bit |
> |---|---|---|
> | 2.0K | 29.7 tok/s | **30.3 tok/s** |
> | 17.4K | 23.6 tok/s | **29.6 tok/s** (+25%) |
>
> For comparison at 17.4K, F16 KV gives 29.2 tok/s and `q8_0/q8_0` KV gives 27.1 tok/s, so the low-bit path is now the fastest of the three while using the smallest cache (614 MiB vs 1429 MiB context buffer). Prefill is unaffected: attention is under 2% of prefill time even at 17K, so prefill keeps converting.
>
> **CUDA limits:** Prefill and query counts above 8 still take a transient FP16 conversion before CUDA FlashAttention. Head sizes other than 128/256, non-NVIDIA devices, or other K/V pairs fall back to the conversion path; unsupported head geometries fall back to CPU. TQ3 weights use dequantization + cuBLAS; eligible small-batch TQ4 weights use a fused decode path. Bonsai's existing CUDA weight kernels are preserved.
>
> **Known stock issue:** the full CUDA `q8_0/q8_0` FlashAttention sweep in `test-backend-ops` aborts with an illegal memory access after several hundred cases, at a case that passes in isolation. No TurboQuant type is involved. Upstream #30103 (a missing barrier in the MMA combine loop) is included, but does not resolve it.
>
> **Model files (Bonsai):**
> - `*-PQ2_0.gguf` (group-128, id 142) — preferred here.
> - `*-Q2_0_g64.gguf` (group-64, id 42) — also loads on mainline llama.cpp.
> - Legacy group-128 stored as id 42 is rejected with a diagnostic.
>
> **Build (Windows example):**
> ```bat
> cmake -S . -B build-vk -G Ninja -DCMAKE_BUILD_TYPE=Release ^
>   -DGGML_VULKAN=ON -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_SERVER=ON
> cmake --build build-vk -j
> ```
> Vulkan needs the [Vulkan SDK](https://vulkan.lunarg.com/) (for `glslc`); set `VULKAN_SDK`.
>
> **Provenance / remotes:** `upstream` = `ggml-org/llama.cpp`, `prism` = `PrismML-Eng/llama.cpp`,
> `turboquant` = `AtomicBot-ai/atomic-llama-cpp-turboquant`, `bt` = `nicheng251/BonsaiTurbo`.
>
> **Status:** experimental, work in progress, **not affiliated** with either upstream. The KV/attention
> rotation is the key integration point: TurboQuant owns KV rotation for `turbo*` cache types, while the
> mainline `attn_rot` path (used by Bonsai) is kept for other quantized KV types — the two must not both
> apply to the same cache.
>
> Do not mix this fork's `ggml-*` libraries with a stock llama.cpp build.

---

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon [In Progress]](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
