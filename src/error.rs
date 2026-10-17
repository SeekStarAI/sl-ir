// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SeekStarAI

//! Error types for IR construction and validation.

use crate::dtype::DType;
use crate::op::OpCode;
use crate::shape::SymId;
use crate::value::{ValueId, ValueRole};
use core::fmt;

/// Failure while constructing or validating an IR object.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IrError {
    /// A value id does not index an existing value.
    UnknownValue(ValueId),
    /// A value id was inserted twice.
    DuplicateValue(ValueId),
    /// A node has the wrong number of inputs or outputs.
    ArityMismatch {
        node: usize,
        expected: usize,
        actual: usize,
    },
    /// A port value has a rank outside the accepted range.
    RankMismatch {
        node: usize,
        port: usize,
        rank: usize,
        min: u8,
        max: u8,
    },
    /// A port does not accept the value's dtype.
    DTypeNotAllowed { node: usize, dtype: DType },
    /// A port does not accept the value's role.
    RoleNotAllowed {
        node: usize,
        value: ValueId,
        role: ValueRole,
    },
    /// A port does not accept symbolic dimensions.
    SymbolicNotAllowed { node: usize },
    /// Element count or byte size overflows `u64`.
    ShapeOverflow,
    /// A dimension does not fit in `u32`.
    DimensionTooLarge(usize),
    /// A quantization group size is zero.
    InvalidGroupSize,
    /// An opaque encoding has no known storage size.
    OpaqueEncodingSize,
    /// A non-parameter value uses a non-dense storage encoding.
    NonDenseValue { value: ValueId, role: ValueRole },
    /// A state value's shape does not match its `StateKind`.
    InvalidStateShape { value: ValueId },
    /// A symbolic dimension survives into a context that requires concrete shapes.
    UnresolvedSymbolicDim(SymId),
    /// A value uses a symbolic dimension that was never declared.
    UndeclaredSymbolicDim(SymId),
    /// The same symbolic dimension is declared twice.
    DuplicateSymbolicDim(SymId),
    /// The requested extent exceeds the declared bound.
    SymbolicDimExceedsMax {
        dim: SymId,
        max: u32,
        requested: u32,
    },
    /// A command buffer contains a symbolic dimension.
    SymbolicDimInCommandBuffer(ValueId),
    /// A value has the wrong role for its position.
    InvalidRole {
        value: ValueId,
        expected: ValueRole,
        actual: ValueRole,
    },
    /// A program input or output has the wrong role.
    InvalidInputOutput { value: ValueId },
    /// A state effect targets a value that has no state declaration.
    StateNotDeclared(ValueId),
    /// A state value is declared twice.
    DuplicateState(ValueId),
    /// A node effect targets a value that is not a state.
    InvalidEffect { node: usize },
    /// A command depends on a command that does not precede it.
    CommandOutOfOrder { command: usize, dependency: u32 },
    /// A memory plan violates an invariant.
    PlanInvariant(&'static str),
    /// Scratchpad space or a slot range overflows.
    PlanOverflow { required: u64, limit: u64 },
    /// A value does not fit in the backend's per-value limit.
    ValueTooLarge {
        value: ValueId,
        bytes: u64,
        limit: u64,
    },
    /// The backend does not implement an operator.
    UnsupportedOperator(OpCode),
    /// The command buffer was produced by a different IR version.
    IrVersionMismatch { found: u32, expected: u32 },
    /// The backend declares a different IR version.
    VersionMismatch { device: u32, ir: u32 },
    /// Capabilities are internally inconsistent.
    InvalidCapabilities(&'static str),
}

impl fmt::Display for IrError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownValue(value) => write!(formatter, "unknown value id {}", value.0),
            Self::DuplicateValue(value) => write!(formatter, "duplicate value id {}", value.0),
            Self::ArityMismatch {
                node,
                expected,
                actual,
            } => write!(
                formatter,
                "node {node} receives {actual} ports; expected {expected}"
            ),
            Self::RankMismatch {
                node,
                port,
                rank,
                min,
                max,
            } => write!(
                formatter,
                "node {node} port {port} has rank {rank}; expected {min}..={max}"
            ),
            Self::DTypeNotAllowed { node, dtype } => {
                write!(formatter, "node {node} does not accept dtype {dtype}")
            }
            Self::RoleNotAllowed { node, value, role } => write!(
                formatter,
                "node {node} does not accept role {role} for value {}",
                value.0
            ),
            Self::SymbolicNotAllowed { node } => {
                write!(formatter, "node {node} does not accept symbolic dimensions")
            }
            Self::ShapeOverflow => formatter.write_str("shape element count overflows u64"),
            Self::DimensionTooLarge(value) => {
                write!(formatter, "dimension {value} does not fit in u32")
            }
            Self::InvalidGroupSize => {
                formatter.write_str("quantization group size must be greater than zero")
            }
            Self::OpaqueEncodingSize => {
                formatter.write_str("opaque quantization has no known storage size")
            }
            Self::NonDenseValue { value, role } => write!(
                formatter,
                "value {} has role {role} but a non-dense storage encoding",
                value.0
            ),
            Self::InvalidStateShape { value } => write!(
                formatter,
                "state value {} does not match its StateKind shape",
                value.0
            ),
            Self::UnresolvedSymbolicDim(dim) => {
                write!(formatter, "symbolic dimension {} is not resolved", dim.0)
            }
            Self::UndeclaredSymbolicDim(dim) => {
                write!(formatter, "symbolic dimension {} is not declared", dim.0)
            }
            Self::DuplicateSymbolicDim(dim) => {
                write!(formatter, "symbolic dimension {} is declared twice", dim.0)
            }
            Self::SymbolicDimExceedsMax {
                dim,
                max,
                requested,
            } => write!(
                formatter,
                "symbolic dimension {} requests {requested} but its bound is {max}",
                dim.0
            ),
            Self::SymbolicDimInCommandBuffer(value) => write!(
                formatter,
                "value {} still contains a symbolic dimension",
                value.0
            ),
            Self::InvalidRole {
                value,
                expected,
                actual,
            } => write!(
                formatter,
                "value {} has role {actual}; expected {expected}",
                value.0
            ),
            Self::InvalidInputOutput { value } => {
                write!(
                    formatter,
                    "value {} has an invalid input/output role",
                    value.0
                )
            }
            Self::StateNotDeclared(value) => {
                write!(formatter, "state {} has no declaration", value.0)
            }
            Self::DuplicateState(value) => {
                write!(formatter, "state {} is declared twice", value.0)
            }
            Self::InvalidEffect { node } => {
                write!(formatter, "node {node} has an effect on a non-state value")
            }
            Self::CommandOutOfOrder {
                command,
                dependency,
            } => write!(
                formatter,
                "command {command} depends on {dependency}, which does not precede it"
            ),
            Self::PlanInvariant(message) => {
                write!(formatter, "memory plan invariant violated: {message}")
            }
            Self::PlanOverflow { required, limit } => {
                write!(
                    formatter,
                    "plan needs {required} bytes but the limit is {limit}"
                )
            }
            Self::ValueTooLarge {
                value,
                bytes,
                limit,
            } => write!(
                formatter,
                "value {} needs {bytes} bytes but the limit is {limit}",
                value.0
            ),
            Self::UnsupportedOperator(op) => write!(formatter, "unsupported operator {op:?}"),
            Self::IrVersionMismatch { found, expected } => write!(
                formatter,
                "command buffer uses IR version {found}; expected {expected}"
            ),
            Self::VersionMismatch { device, ir } => write!(
                formatter,
                "backend declares IR version {device}; compiler uses {ir}"
            ),
            Self::InvalidCapabilities(message) => {
                write!(formatter, "invalid device capabilities: {message}")
            }
        }
    }
}

impl core::error::Error for IrError {}

#[cfg(test)]
mod tests {
    use super::IrError;
    use crate::shape::SymId;
    use alloc::string::ToString;

    #[test]
    fn errors_render_actionable_messages() {
        assert_eq!(
            IrError::SymbolicDimExceedsMax {
                dim: SymId(1),
                max: 64,
                requested: 128,
            }
            .to_string(),
            "symbolic dimension 1 requests 128 but its bound is 64"
        );
        assert_eq!(
            IrError::PlanInvariant("slot exceeds scratchpad").to_string(),
            "memory plan invariant violated: slot exceeds scratchpad"
        );
    }
}
