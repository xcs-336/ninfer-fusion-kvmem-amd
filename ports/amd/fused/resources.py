"""Physical LDS admission for fused integer-A8 and grouped Q8 launch closures."""
from __future__ import annotations

import re


def transform(relative: str, text: str, helpers) -> str:
    a8_calls = {
        "linear_swiglu/q4a8/q5a8_linear_add.cu":
            ("launch", "kCols", "launch<kCols, NT / 2>(x, down, residual, tokens, codes, scales, stream);"),
        "gdn_input_proj/q4_q5/q4_q5_gdn_input_a8.cu":
            ("launch_all", "kHidden", "launch_all<NT / 2>(x, qk, value_z, tokens, codes, scales, qkv, z, stream);"),
        "attn_input_proj/q4_q5/q4_q5_attn_input_a8.cu":
            ("launch_all", "kHidden", "launch_all<NT / 2>(x, query_key, gate_value, tokens, codes, scales, q, gate, k, v, stream);"),
    }
    for suffix, (function, hidden, invocation) in a8_calls.items():
        if not relative.endswith(suffix): continue
        def adapt_a8(definition: str) -> str:
            opening = definition.index("{")
            body = definition[opening + 1:-1]
            # Q5's extra high plane dominates Q4 for the shared row map. The
            # quantizer owns fragment-order activation storage, so re-enter the
            # entire quantize+consumer closure with one admitted NT throughout.
            guard = f"""
    static_assert(NT > 0 && (NT & (NT - 1)) == 0,
        "native integer-A8 token partition must be a positive power of two");
    constexpr std::size_t NativeBytes = a8::shared_bytes<a8::Q5Codec, 1, NT, Rows>({hidden});
    if constexpr (NativeBytes > 65536) {{
        static_assert(NT > 1, "native integer-A8 minimum tile exceeds 64-KiB LDS");
        {invocation}
        return;
    }} else {{
"""
            return definition[:opening + 1] + guard + body + "\n    }\n}"
        text = helpers.definition(text, rf"void {function}\([^{{}}]+\)\s*\{{", adapt_a8)
    names = {
        "linear_add/q8/q8_linear_add_gemm_splitk.cu": ("launch_medium",),
        "linear_add/q8/q8_linear_add_gemm_splitk_unified.cu": ("launch_medium",),
        "linear_pair/q8/q8_pair_gemm_splitk.cu": ("launch_medium",),
        "linear_pair/q8/q8_pair_gemm_splitk_unified.cu": ("launch_medium",),
        "gdn_input_proj/q8/q8_gdn_input_gemm_splitk_unified.cu": ("launch_medium_cols",),
        "attn_input_proj/q8/q8_attn_input_gemm_splitk.cu":
            ("launch_target_medium_cols", "launch_companion_medium_cols"),
        "attn_input_proj/q8/q8_attn_input_gemm_splitk_unified.cu":
            ("launch_target_medium_cols", "launch_companion_medium_cols"),
    }
    for suffix, functions in names.items():
        if not relative.endswith(suffix): continue
        if "/linear_pair/" in "/" + relative and "#include <algorithm>" not in text:
            text = "#include <algorithm>\n" + text
        for function in functions:
            def adapt(definition: str) -> str:
                opening = definition.index("{")
                original = definition[opening + 1:-1]
                # Both legacy allocations and the unified grouped Schedule
                # use code[16][KSplits*64] and
                # BF16 activation[KSplits*NGroups][(TileCols/NGroups)*64].
                # Its FP32 partials reuse the activation allocation.
                admitted = """
    constexpr int native_per_split = 1024 + 128 * TileCols;
    constexpr int NativeKSplits = KSplits * native_per_split <= 65536 ? KSplits :
        (KSplits >= 4 && 4 * native_per_split <= 65536 ? 4 : 2);
    static_assert(NativeKSplits * native_per_split <= 65536,
        "Native gfx1201 fused Q8 grouped allocation exceeds 64-KiB LDS");
"""
                original = re.sub(r"\bKSplits\b", "NativeKSplits", original)
                if relative.endswith(("linear_pair/q8/q8_pair_gemm_splitk.cu",
                                      "linear_pair/q8/q8_pair_gemm_splitk_unified.cu")):
                    # C256 cannot fit even at the kernel's minimum two K
                    # partitions (67,584 bytes). Token intervals are independent.
                    # Existing Tensor views preserve all original weight/output
                    # bytes and require no extra arena allocation or repacking.
                    chunk = """
    if constexpr (2 * (1024 + 128 * TileCols) > 65536) {
        static_assert(TileCols % 2 == 0 && (TileCols / 2) % NGroups == 0 &&
            ((TileCols / 2) / NGroups) % 8 == 0);
        for (std::int32_t offset = 0; offset < x.ne[1];) {
            const std::int32_t count = std::min<std::int32_t>(TileCols / 2, x.ne[1] - offset);
            const Tensor x_slice = x.slice(1, offset, count);
            Tensor first_slice = first_out.slice(1, offset, count);
            Tensor second_slice = second_out.slice(1, offset, count);
            launch_medium<TileCols / 2, KSplits, NGroups, MinBlocks>(
                x_slice, first_weight, second_weight, first_slice, second_slice, stream);
            offset += count;
        }
        return;
    } else {
"""
                    return definition[:opening + 1] + chunk + admitted + original + "\n    }\n}"
                return definition[:opening + 1] + admitted + original + "\n}"
            text = helpers.definition(text, rf"void {function}\([^{{}}]+\)\s*\{{", adapt)
    return text
