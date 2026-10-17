// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SeekStarAI

//! Operator vocabulary, port signatures and operator sets.

use crate::dtype::DTypeSet;
use crate::value::ValueRoleSet;
use alloc::vec::Vec;

/// Stable operator identifier.
///
/// Discriminants are part of the ABI and are never reused. Removed operators
/// keep their value as a tombstone.
#[repr(u16)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum OpCode {
    /// Batched `C = A * B^T` on trailing dimensions.
    MatMulNT = 0,
    /// Batched `C = A * B` on trailing dimensions.
    MatMulNN = 1,
    /// Softmax along the last dimension.
    Softmax = 2,
    /// Causal softmax along the last dimension.
    SoftmaxCausal = 3,
    /// Root-mean-square normalization.
    RmsNorm = 4,
    /// Rotary position embedding.
    Rope = 5,
    /// SiLU activation.
    Silu = 6,
    /// Elementwise multiplication.
    Mul = 7,
    /// Elementwise addition.
    Add = 8,
    /// Elementwise scaling.
    Scale = 9,
    /// Row gathering from a table.
    Gather = 10,
    /// Selection of the last position of each batch row.
    LastToken = 11,
    /// KV-cache append with a state write effect.
    KvAppend = 12,
    /// `[B, M, H*D] -> [B, H, M, D]`.
    SplitHeads = 13,
    /// `[B, H, M, D] -> [B, M, H*D]`.
    MergeHeads = 14,
    /// Fused causal attention over a KV cache.
    CausalAttention = 15,
    /// Fused SwiGLU feed-forward block.
    SwiGlu = 16,
    /// Fusion hint: RMSNorm followed by a matmul.
    RmsNormMatMul = 17,
}

impl OpCode {
    /// Returns whether the operator is optional and may be decomposed.
    pub const fn is_composite(self) -> bool {
        matches!(
            self,
            Self::CausalAttention | Self::SwiGlu | Self::RmsNormMatMul
        )
    }

    /// Returns whether every backend must declare support explicitly.
    pub const fn is_core(self) -> bool {
        !self.is_composite()
    }

    /// Returns the port signature of the operator.
    pub const fn signature(self) -> Signature {
        match self {
            Self::MatMulNT => SIG_MATMUL_NT,
            Self::MatMulNN => SIG_MATMUL_NN,
            Self::Softmax => SIG_SOFTMAX,
            Self::SoftmaxCausal => SIG_SOFTMAX_CAUSAL,
            Self::RmsNorm => SIG_RMS_NORM,
            Self::Rope => SIG_ROPE,
            Self::Silu => SIG_SILU,
            Self::Mul => SIG_MUL,
            Self::Add => SIG_ADD,
            Self::Scale => SIG_SCALE,
            Self::Gather => SIG_GATHER,
            Self::LastToken => SIG_LAST_TOKEN,
            Self::KvAppend => SIG_KV_APPEND,
            Self::SplitHeads => SIG_SPLIT_HEADS,
            Self::MergeHeads => SIG_MERGE_HEADS,
            Self::CausalAttention => SIG_CAUSAL_ATTENTION,
            Self::SwiGlu => SIG_SWI_GLU,
            Self::RmsNormMatMul => SIG_RMS_NORM_MATMUL,
        }
    }
}

/// Rotary position embedding layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RopeVariant {
    /// Split-half rotation used by Llama-style models.
    Neox,
    /// Interleaved rotation used by GPT-J-style models.
    GptJ,
}

/// Operator instance with its constant attributes.
#[derive(Clone, Copy, Debug)]
pub enum Op {
    /// Batched matmul with a transposed right-hand side.
    MatMulNT,
    /// Batched matmul.
    MatMulNN,
    /// Softmax along the last dimension.
    Softmax,
    /// Causal softmax along the last dimension.
    SoftmaxCausal,
    /// RMSNorm with the given epsilon.
    RmsNorm { eps: f32 },
    /// Rotary position embedding.
    Rope { variant: RopeVariant, theta: f32 },
    /// SiLU activation.
    Silu,
    /// Elementwise multiplication.
    Mul,
    /// Elementwise addition.
    Add,
    /// Elementwise scaling.
    Scale { factor: f32 },
    /// Row gather.
    Gather,
    /// Last-position selection.
    LastToken,
    /// KV append for one layer.
    KvAppend { layer: u32 },
    /// Head split for one attention layer.
    SplitHeads { heads: u32, head_dim: u32 },
    /// Head merge for one attention layer.
    MergeHeads { heads: u32, head_dim: u32 },
    /// Fused causal attention.
    CausalAttention {
        heads: u32,
        kv_heads: u32,
        head_dim: u32,
        scale: f32,
    },
    /// Fused SwiGLU.
    SwiGlu,
    /// RMSNorm + matmul fusion hint.
    RmsNormMatMul { eps: f32 },
}

impl Op {
    /// Returns the stable operator code.
    pub const fn code(&self) -> OpCode {
        match self {
            Self::MatMulNT => OpCode::MatMulNT,
            Self::MatMulNN => OpCode::MatMulNN,
            Self::Softmax => OpCode::Softmax,
            Self::SoftmaxCausal => OpCode::SoftmaxCausal,
            Self::RmsNorm { .. } => OpCode::RmsNorm,
            Self::Rope { .. } => OpCode::Rope,
            Self::Silu => OpCode::Silu,
            Self::Mul => OpCode::Mul,
            Self::Add => OpCode::Add,
            Self::Scale { .. } => OpCode::Scale,
            Self::Gather => OpCode::Gather,
            Self::LastToken => OpCode::LastToken,
            Self::KvAppend { .. } => OpCode::KvAppend,
            Self::SplitHeads { .. } => OpCode::SplitHeads,
            Self::MergeHeads { .. } => OpCode::MergeHeads,
            Self::CausalAttention { .. } => OpCode::CausalAttention,
            Self::SwiGlu => OpCode::SwiGlu,
            Self::RmsNormMatMul { .. } => OpCode::RmsNormMatMul,
        }
    }

    /// Returns the layer index for stateful operators.
    pub const fn layer(&self) -> Option<u32> {
        match self {
            Self::KvAppend { layer } => Some(*layer),
            _ => None,
        }
    }
}

/// Requirements for one operator port.
#[derive(Clone, Copy, Debug)]
pub struct PortSpec {
    /// Accepted element types.
    pub dtypes: DTypeSet,
    /// Accepted value roles.
    pub roles: ValueRoleSet,
    /// Minimum rank.
    pub min_rank: u8,
    /// Maximum rank.
    pub max_rank: u8,
    /// Whether symbolic dimensions are accepted.
    pub allow_symbolic: bool,
}

impl PortSpec {
    /// A floating-point port.
    pub const fn f32(roles: ValueRoleSet, min_rank: u8, max_rank: u8) -> Self {
        Self {
            dtypes: DTypeSet::ONLY_F32,
            roles,
            min_rank,
            max_rank,
            allow_symbolic: true,
        }
    }

    /// An unsigned integer port.
    pub const fn u32(roles: ValueRoleSet, min_rank: u8, max_rank: u8) -> Self {
        Self {
            dtypes: DTypeSet::ONLY_U32,
            roles,
            min_rank,
            max_rank,
            allow_symbolic: true,
        }
    }
}

/// Input and output ports of an operator.
#[derive(Clone, Copy, Debug)]
pub struct Signature {
    /// Input ports in call order.
    pub inputs: &'static [PortSpec],
    /// Output ports in binding order.
    pub outputs: &'static [PortSpec],
}

const R: ValueRoleSet = ValueRoleSet::READABLE;
const W: ValueRoleSet = ValueRoleSet::WRITABLE;
const P: ValueRoleSet = ValueRoleSet::PARAM;
const I: ValueRoleSet = ValueRoleSet::INPUT;
const S: ValueRoleSet = ValueRoleSet::SCALAR;

const SIG_MATMUL_NT: Signature = Signature {
    inputs: &[PortSpec::f32(R, 2, 4), PortSpec::f32(R, 2, 4)],
    outputs: &[PortSpec::f32(W, 2, 4)],
};

const SIG_MATMUL_NN: Signature = Signature {
    inputs: &[PortSpec::f32(R, 2, 4), PortSpec::f32(R, 2, 4)],
    outputs: &[PortSpec::f32(W, 2, 4)],
};

const SIG_SOFTMAX: Signature = Signature {
    inputs: &[PortSpec::f32(R, 1, 4)],
    outputs: &[PortSpec::f32(W, 1, 4)],
};

const SIG_SOFTMAX_CAUSAL: Signature = Signature {
    inputs: &[
        PortSpec::f32(R, 2, 4),
        PortSpec::u32(S, 0, 0),
        PortSpec::u32(S, 0, 0),
    ],
    outputs: &[PortSpec::f32(W, 2, 4)],
};

const SIG_RMS_NORM: Signature = Signature {
    inputs: &[PortSpec::f32(R, 1, 4), PortSpec::f32(P, 1, 1)],
    outputs: &[PortSpec::f32(W, 1, 4)],
};

const SIG_ROPE: Signature = Signature {
    inputs: &[PortSpec::f32(R, 3, 4), PortSpec::u32(I, 2, 2)],
    outputs: &[PortSpec::f32(W, 3, 4)],
};

const SIG_SILU: Signature = Signature {
    inputs: &[PortSpec::f32(R, 1, 4)],
    outputs: &[PortSpec::f32(W, 1, 4)],
};

const SIG_MUL: Signature = Signature {
    inputs: &[PortSpec::f32(R, 1, 4), PortSpec::f32(R, 1, 4)],
    outputs: &[PortSpec::f32(W, 1, 4)],
};

const SIG_ADD: Signature = Signature {
    inputs: &[PortSpec::f32(R, 1, 4), PortSpec::f32(R, 1, 4)],
    outputs: &[PortSpec::f32(W, 1, 4)],
};

const SIG_SCALE: Signature = Signature {
    inputs: &[PortSpec::f32(R, 1, 4)],
    outputs: &[PortSpec::f32(W, 1, 4)],
};

const SIG_GATHER: Signature = Signature {
    inputs: &[PortSpec::f32(P, 2, 2), PortSpec::u32(I, 1, 3)],
    outputs: &[PortSpec::f32(W, 2, 4)],
};

const SIG_LAST_TOKEN: Signature = Signature {
    inputs: &[PortSpec::f32(R, 3, 3), PortSpec::u32(S, 0, 0)],
    outputs: &[PortSpec::f32(W, 2, 2)],
};

const SIG_KV_APPEND: Signature = Signature {
    inputs: &[
        PortSpec::f32(R, 4, 4),
        PortSpec::f32(R, 4, 4),
        PortSpec::u32(S, 0, 0),
    ],
    outputs: &[],
};

const SIG_SPLIT_HEADS: Signature = Signature {
    inputs: &[PortSpec::f32(R, 3, 3)],
    outputs: &[PortSpec::f32(W, 4, 4)],
};

const SIG_MERGE_HEADS: Signature = Signature {
    inputs: &[PortSpec::f32(R, 4, 4)],
    outputs: &[PortSpec::f32(W, 3, 3)],
};

const SIG_CAUSAL_ATTENTION: Signature = Signature {
    inputs: &[
        PortSpec::f32(R, 4, 4),
        PortSpec::f32(R, 4, 4),
        PortSpec::f32(R, 4, 4),
        PortSpec::u32(S, 0, 0),
    ],
    outputs: &[PortSpec::f32(W, 4, 4)],
};

const SIG_SWI_GLU: Signature = Signature {
    inputs: &[
        PortSpec::f32(R, 2, 4),
        PortSpec::f32(P, 2, 2),
        PortSpec::f32(P, 2, 2),
        PortSpec::f32(P, 2, 2),
    ],
    outputs: &[PortSpec::f32(W, 2, 4)],
};

const SIG_RMS_NORM_MATMUL: Signature = Signature {
    inputs: &[
        PortSpec::f32(R, 2, 4),
        PortSpec::f32(P, 1, 1),
        PortSpec::f32(P, 2, 2),
    ],
    outputs: &[PortSpec::f32(W, 2, 4)],
};

/// Sorted set of operator codes.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct OpSet(Vec<OpCode>);

impl OpSet {
    /// Creates a set, sorting and deduplicating the input.
    pub fn new(mut ops: Vec<OpCode>) -> Self {
        ops.sort_unstable();
        ops.dedup();
        Self(ops)
    }

    /// Returns whether the set contains `op`.
    pub fn contains(&self, op: OpCode) -> bool {
        self.0.binary_search(&op).is_ok()
    }

    /// Returns the operators in ascending order.
    pub fn as_slice(&self) -> &[OpCode] {
        &self.0
    }

    /// Returns an iterator over the set.
    pub fn iter(&self) -> impl Iterator<Item = &OpCode> {
        self.0.iter()
    }

    /// Returns the number of operators.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns whether the set is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl<'a> IntoIterator for &'a OpSet {
    type Item = &'a OpCode;
    type IntoIter = core::slice::Iter<'a, OpCode>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::{Op, OpCode, OpSet, RopeVariant};
    use crate::value::ValueRole;
    use alloc::vec;

    #[test]
    fn op_codes_round_trip() {
        let op = Op::RmsNorm { eps: 1e-5 };
        assert_eq!(op.code(), OpCode::RmsNorm);
        assert_eq!(Op::KvAppend { layer: 3 }.layer(), Some(3));
        assert_eq!(Op::Silu.layer(), None);
        assert_eq!(
            Op::Rope {
                variant: RopeVariant::Neox,
                theta: 10_000.0,
            }
            .code(),
            OpCode::Rope
        );
    }

    #[test]
    fn signatures_expose_shapes_and_roles() {
        let matmul = OpCode::MatMulNT.signature();
        assert_eq!(matmul.inputs.len(), 2);
        assert_eq!(matmul.outputs.len(), 1);
        assert!(matmul.inputs[0].roles.contains(ValueRole::Param));
        assert!(!matmul.inputs[0].roles.contains(ValueRole::Output));

        let attention = OpCode::CausalAttention.signature();
        assert_eq!(attention.inputs.len(), 4);
        assert_eq!(attention.inputs[3].max_rank, 0);
        assert!(OpCode::KvAppend.signature().outputs.is_empty());
    }

    #[test]
    fn composite_classification_is_explicit() {
        assert!(OpCode::CausalAttention.is_composite());
        assert!(OpCode::SwiGlu.is_composite());
        assert!(OpCode::MatMulNT.is_core());
        assert!(!OpCode::MatMulNT.is_composite());
    }

    #[test]
    fn op_sets_sort_and_deduplicate() {
        let set = OpSet::new(vec![OpCode::Silu, OpCode::MatMulNT, OpCode::Silu]);
        assert_eq!(set.len(), 2);
        assert!(set.contains(OpCode::MatMulNT));
        assert!(!set.contains(OpCode::Add));
        assert_eq!(set.as_slice(), [OpCode::MatMulNT, OpCode::Silu].as_slice());
        assert_eq!(set.iter().count(), 2);
    }
}
