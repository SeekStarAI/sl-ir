// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SeekStarAI

//! Device capabilities negotiated at graph-compile time.

use crate::dtype::DTypeSet;
use crate::error::IrError;
use crate::op::{OpCode, OpSet};
use crate::shape::ShapeConstraints;
use crate::IR_VERSION;

/// Human-readable backend identifier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BackendTag(pub &'static str);

/// What a backend can execute.
#[derive(Clone, Debug)]
pub struct DeviceCapabilities {
    /// Backend identifier.
    pub backend: BackendTag,
    /// IR version the backend was built against.
    pub ir_version: u32,
    /// Supported mandatory operators.
    pub core_ops: OpSet,
    /// Supported optional operators.
    pub composite_ops: OpSet,
    /// Supported element types.
    pub dtypes: DTypeSet,
    /// Guaranteed buffer alignment in bytes; must be at least 64.
    pub min_buffer_align: u32,
    /// Largest scratchpad the backend accepts.
    pub max_scratchpad_bytes: u64,
    /// Largest single value the backend accepts.
    pub max_value_bytes: u64,
    /// Backend-specific shape constraints.
    pub constraints: ShapeConstraints,
}

impl DeviceCapabilities {
    /// Returns whether the backend implements `op`, natively or at all.
    pub fn supports(&self, op: OpCode) -> bool {
        if op.is_composite() {
            self.composite_ops.contains(op)
        } else {
            self.core_ops.contains(op)
        }
    }

    /// Validates self-consistency.
    pub fn validate(&self) -> Result<(), IrError> {
        if self.ir_version != IR_VERSION {
            return Err(IrError::VersionMismatch {
                device: self.ir_version,
                ir: IR_VERSION,
            });
        }
        if self.min_buffer_align < 64 {
            return Err(IrError::InvalidCapabilities(
                "min_buffer_align must be at least 64",
            ));
        }
        if self.dtypes.is_empty() {
            return Err(IrError::InvalidCapabilities("dtypes must not be empty"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{BackendTag, DeviceCapabilities};
    use crate::dtype::DTypeSet;
    use crate::op::{OpCode, OpSet};
    use crate::shape::ShapeConstraints;
    use crate::IR_VERSION;
    use alloc::vec;

    fn capabilities() -> DeviceCapabilities {
        DeviceCapabilities {
            backend: BackendTag("unit-test"),
            ir_version: IR_VERSION,
            core_ops: OpSet::new(vec![OpCode::MatMulNT, OpCode::RmsNorm]),
            composite_ops: OpSet::new(vec![OpCode::CausalAttention]),
            dtypes: DTypeSet::ONLY_F32,
            min_buffer_align: 64,
            max_scratchpad_bytes: 1 << 30,
            max_value_bytes: 1 << 28,
            constraints: ShapeConstraints::default(),
        }
    }

    #[test]
    fn supports_separates_core_from_composite() {
        let caps = capabilities();
        assert!(caps.supports(OpCode::MatMulNT));
        assert!(!caps.supports(OpCode::Softmax));
        assert!(caps.supports(OpCode::CausalAttention));
        assert!(!caps.supports(OpCode::SwiGlu));
    }

    #[test]
    fn rejects_inconsistent_capabilities() {
        let mut caps = capabilities();
        caps.ir_version = IR_VERSION + 1;
        assert!(caps.validate().is_err());

        let mut caps = capabilities();
        caps.min_buffer_align = 32;
        assert!(caps.validate().is_err());
    }
}
