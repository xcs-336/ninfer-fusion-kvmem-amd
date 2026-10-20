The native Flash-Next prefill launcher composes the existing public
`selected_block_attention` Op. Its original `use_mma=true` default already used
that same public Op; the alternate NVIDIA schedule required 80,064 bytes of LDS.
The original manifest still contains its source TU and private kernel definitions.
The private schedule has no native runtime qualification.

| Contract | Model operand / behavior | Public selected-block operand / behavior |
|---|---|---|
| Query | `scratch.query`, BF16 `[256,24,tokens]` | Same tensor or contiguous token slice |
| Scale/GQA | `1/16`, query head `h` reads KV head `h/12` | Same fixed profile |
| Positions | Inclusive absolute I32 `token_indices` | Same positions, no renumbering |
| Complete blocks | Four-token blocks from `selected_blocks [512,tokens]` | Same indices, order and counts |
| Tail | `[4*floor((p+1)/4),p]` | Same interval; empty visible set produces exact zero |
| Page mapping | I32 `[logical_pages,table_rows]`, shared `table_row` | Same table and row |
| K/V | `[256,64,2,physical_pages]`, BF16 or unscaled OCP E4M3FN | Same represented bytes |
| Output | BF16 `scratch.attended [256,24,tokens]` | Same output tensor or token slice |
| Other casts | Query/key RMSNorm, RoPE, persistent storage casts | Original prepare/store kernels retained verbatim |
| Gate | BF16 `attended * sigmoid(gate)` | Original gate kernel retained verbatim |
| Lifetime | Caller-owned scratch/cache, same stream; no Op allocation | Same ownership and stream |
| Large token counts | Original positive token domain | Split only at 262,144 queries per public Op launch |

The model's existing host validation checks query/output geometry, cache types,
page-table geometry, table row, contiguity and 16-byte alignment. The same tensor
views reach the public Op's own checks. Slicing only the token axis preserves
absolute positions, complete selected rows and page identity, including the final
partial chunk. No tensor content is repacked or converted by this composition.

The unused experimental MMA schedule additionally rounded its transient
softmax tile to BF16. The native public route uses FP32 softmax/PV and the same
observable BF16 `attended` boundary; this is also the original launcher's public
Op route. The experimental intermediate approximation is not a stored artifact
format or an emitted tensor, and no native qualification is claimed for it.

Public selected-block qualification uses its actual production implementation
against a naive FP64 attention oracle, both storage types, shared/batched page
rows, sparse/dense/empty visibility, page boundaries, graph capture, immutable
inputs and output byte guards. Those results qualify the reused Op. They do not
qualify the unused private MMA schedule, the unchanged QSA preparation/store/gate
kernels, or end-to-end model execution; the latter still needs Engine integration
tests and an actual `.ninfer` artifact.
