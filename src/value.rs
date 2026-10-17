// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SeekStarAI

//! Values, their roles and role sets.

use crate::dtype::DType;
use crate::error::IrError;
use crate::shape::Shape;
use core::fmt;

/// Index of a value in a program or command buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ValueId(pub u32);

/// Role a value plays in the graph.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ValueRole {
    /// Per-step input such as token ids or positions.
    Input = 0,
    /// Per-step integer scalar such as `seq_len` or `past_len`.
    Scalar = 1,
    /// Per-step result such as logits.
    Output = 2,
    /// Persistent weight bound at compile time.
    Param = 3,
    /// Persistent state such as the KV cache.
    State = 4,
    /// Static scratchpad allocation.
    Temp = 5,
}

impl ValueRole {
    /// Number of defined roles.
    pub const COUNT: usize = 6;

    /// Returns the stable bit index used by [`ValueRoleSet`].
    pub const fn index(self) -> usize {
        self as usize
    }
}

impl fmt::Display for ValueRole {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Input => "Input",
            Self::Scalar => "Scalar",
            Self::Output => "Output",
            Self::Param => "Param",
            Self::State => "State",
            Self::Temp => "Temp",
        })
    }
}

/// Compact set of value roles.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ValueRoleSet(u8);

impl ValueRoleSet {
    /// Empty set.
    pub const EMPTY: Self = Self(0);

    /// Every role.
    pub const ANY: Self = Self((1u8 << ValueRole::COUNT) - 1);

    /// Role `Input`.
    pub const INPUT: Self = Self::of(ValueRole::Input);

    /// Role `Scalar`.
    pub const SCALAR: Self = Self::of(ValueRole::Scalar);

    /// Role `Output`.
    pub const OUTPUT: Self = Self::of(ValueRole::Output);

    /// Role `Param`.
    pub const PARAM: Self = Self::of(ValueRole::Param);

    /// Role `State`.
    pub const STATE: Self = Self::of(ValueRole::State);

    /// Role `Temp`.
    pub const TEMP: Self = Self::of(ValueRole::Temp);

    /// Roles accepted by an operator input: persistent weights, scratchpad
    /// values and per-step inputs.
    pub const READABLE: Self = Self::PARAM.union(Self::TEMP).union(Self::INPUT);

    /// Roles an operator may write: scratchpad values and bound outputs.
    pub const WRITABLE: Self = Self::TEMP.union(Self::OUTPUT);

    /// Creates a set containing a single role.
    pub const fn of(role: ValueRole) -> Self {
        Self(1u8 << (role.index() as u8))
    }

    /// Returns the union of two sets.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Returns whether the set contains `role`.
    pub const fn contains(self, role: ValueRole) -> bool {
        self.0 & (1u8 << (role.index() as u8)) != 0
    }

    /// Returns whether the set is empty.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// A typed and shaped IR value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Value {
    /// Element type.
    pub dtype: DType,
    /// Logical shape; may contain symbolic dimensions in a `Program`.
    pub shape: Shape,
    /// Role the value plays.
    pub role: ValueRole,
}

impl Value {
    /// Creates a value.
    pub fn new(dtype: DType, shape: Shape, role: ValueRole) -> Self {
        Self { dtype, shape, role }
    }

    /// Returns the storage size in bytes.
    pub fn byte_size(&self) -> Result<u64, IrError> {
        self.shape.byte_size(self.dtype)
    }

    /// Rejects the value unless it has the expected role.
    pub fn require_role(&self, expected: ValueRole) -> Result<(), IrError> {
        if self.role == expected {
            Ok(())
        } else {
            Err(IrError::InvalidRole {
                value: ValueId(0),
                expected,
                actual: self.role,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Value, ValueId, ValueRole, ValueRoleSet};
    use crate::{DType, Shape};

    #[test]
    fn role_sets_are_queryable() {
        assert!(ValueRoleSet::READABLE.contains(ValueRole::Param));
        assert!(!ValueRoleSet::READABLE.contains(ValueRole::State));
        assert!(ValueRoleSet::WRITABLE.contains(ValueRole::Output));
        assert!(ValueRoleSet::ANY.contains(ValueRole::Scalar));
        assert!(ValueRoleSet::EMPTY.is_empty());
    }

    #[test]
    fn values_report_byte_sizes() {
        let value = Value::new(DType::F32, Shape::consts(&[2, 3]), ValueRole::Temp);
        assert_eq!(value.byte_size().unwrap(), 24);
        assert_eq!(value.role, ValueRole::Temp);
        assert_eq!(ValueId(0).0, 0);
    }

    #[test]
    fn scalar_has_no_dimensions() {
        let scalar = Value::new(DType::U32, Shape::default(), ValueRole::Scalar);
        assert_eq!(scalar.byte_size().unwrap(), 4);
        assert!(scalar.shape.as_slice().is_empty());
        assert!(ValueRoleSet::ANY.contains(ValueRole::Output));
    }
}
