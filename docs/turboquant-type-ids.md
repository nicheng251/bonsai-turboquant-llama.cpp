# ggml_type id allocation

This fork assigns explicit ids to types that upstream llama.cpp does not have.
The ids are part of the GGUF file format for the weight types, so they cannot be
changed without re-converting the affected files.

## Current allocation

| id | type | kind | appears in a GGUF file |
|---|---|---|---|
| 43 | `GGML_TYPE_TURBO2_0` | TurboQuant 2-bit KV cache | no, runtime-only |
| 44 | `GGML_TYPE_TURBO3_0` | TurboQuant 3-bit KV cache | no, runtime-only |
| 45 | `GGML_TYPE_TQ3_1S` | TurboQuant 3-bit weights | yes |
| 46 | `GGML_TYPE_TQ4_1S` | TurboQuant 4-bit weights | yes |
| 47 | `GGML_TYPE_TURBO4_0` | TurboQuant 4-bit KV cache | no, runtime-only |
| 48 | `GGML_TYPE_Q8_CR` | ConvRot Q8_0 weights | yes |
| 49 | `GGML_TYPE_Q5_CR` | ConvRot Q5_0 weights | yes |
| 50 | `GGML_TYPE_Q6_CR` | ConvRot Q6_K weights | yes |
| 142 | `GGML_TYPE_PQ2_0` | Prism ternary, group 128 | yes |
| 143 | `GGML_TYPE_PTQ1_0` | Prism ternary weights, group 128 | yes |
| 144 | `GGML_TYPE_COUNT` | sentinel | - |

Upstream allocated 0..42 when these ids were chosen; its last type is
`GGML_TYPE_Q2_0 = 42`. Ids 3-5, 31-33 and 36-38 are retired upstream ids and are
not reused.

## Why 43 is a risk

Upstream appends new types sequentially, so the next upstream type takes id 43.
Two enumerators with the same value are legal C++: the build stays green,
`type_traits[43]` resolves to whichever initializer is applied last, and a model
file that stores type 43 is then read with the wrong block layout. This is a
silent data-corruption class of bug, not a compile error.

`scripts/check-ggml-type-ids.py` turns it into a build-time check. It fails when

- two enumerators share an id,
- an id is at or past `GGML_TYPE_COUNT`,
- a type that is not in the fork's private table has an id inside the private
  range, or
- a private type moved from the id listed above.

Run it in CI and before every rebase on upstream:

```sh
python3 scripts/check-ggml-type-ids.py
```

## Proposed target allocation

Use this when the checker reports a collision. The numbers stay far above the
upstream range, and the retired gaps are left alone.

| now | proposed | type |
|---|---|---|
| 43 | 160 | `GGML_TYPE_TURBO2_0` |
| 44 | 161 | `GGML_TYPE_TURBO3_0` |
| 45 | 162 | `GGML_TYPE_TQ3_1S` |
| 46 | 163 | `GGML_TYPE_TQ4_1S` |
| 47 | 164 | `GGML_TYPE_TURBO4_0` |
| 48 | 165 | `GGML_TYPE_Q8_CR` |
| 49 | 166 | `GGML_TYPE_Q5_CR` |
| 50 | 167 | `GGML_TYPE_Q6_CR` |
| 142 | 142 | `GGML_TYPE_PQ2_0` (unchanged, Prism model files) |
| 143 | 143 | `GGML_TYPE_PTQ1_0` (unchanged, Prism model files) |
| 144 | 168 | `GGML_TYPE_COUNT` |

## Renumbering procedure

Do this only when the checker reports a collision. Weight ids are cheap to move
today because no file with `TQ3_1S`, `TQ4_1S`, `Q8_CR`, `Q5_CR` or `Q6_CR` has
been published.

1. Choose the new ids. Keep them above the current upstream end of the enum and
   away from the retired-id gaps. Leave `PQ2_0` / `PTQ1_0` at 142/143: Prism's
   Bonsai model files depend on those values.
2. Edit `ggml/include/ggml.h` only. Every other use is by enumerator name; the
   backends' `type_traits` tables are indexed by the enum and sized by
   `GGML_TYPE_COUNT`.
3. Update `scripts/check-ggml-type-ids.py` (`PRIVATE`) and the table above.
4. Rebuild CPU, CUDA and Vulkan and run:
   - `test-turbo-quant`
   - `test-gguf`
   - `test-backend-ops -o MUL_MAT -p tq`
   - `test-backend-ops -o FLASH_ATTN_EXT -p "kv=113"`
5. If a GGUF with a moved weight type exists anywhere, re-convert it. KV cache
   types never enter a file, so they carry no migration cost.

## Related

- Type sizes and block layouts: `ggml/include/ggml.h`, `ggml/src/ggml-turbo-quant.c`,
  `ggml/src/ggml-cuda/turbo-quant.cuh`.
- Backend coverage: `docs/bonsai-turboquant.md`.
