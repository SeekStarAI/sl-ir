// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SeekStarAI

//! Values, their roles, storage encodings and role sets.

use crate::dtype::DType;
use crate::error::IrError;
use crate::shape::Shape;
use alloc::string::String;
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

/// Integer packing format of a quantized tensor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuantizationFormat {
    /// Signed 4-bit payload packed two values per byte.
    Int4,
    /// Signed 8-bit payload.
    Int8,
}

impl QuantizationFormat {
    /// Returns the storage width of one quantized value in bits.
    pub const fn bits(self) -> u8 {
        match self {
            Self::Int4 => 4,
            Self::Int8 => 8,
        }
    }
}

/// Physical storage encoding of a value.
///
/// [`Value::dtype`] is the **logical** element type; this descriptor says how
/// the bytes are stored. Only `Param` values may be quantized; activations,
/// inputs, outputs, states and temporaries must be [`Self::Dense`].
///
/// Quantized payloads carry the packed data only. Scale tensors are separate
/// values (for example `<weight>.__scales`), so they stay ordinary F32 params.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TensorEncoding {
    /// Dense elements of `Value::dtype`, row-major.
    Dense,
    /// Group-wise symmetric quantization of a logical tensor.
    GroupwiseQuantized {
        /// Integer packing format.
        format: QuantizationFormat,
        /// Number of logical elements represented by one scale.
        group_size: u32,
    },
    /// A quantization scheme this IR does not interpret.
    OpaqueQuantized {
        /// Scheme identifier preserved from the source format.
        scheme: String,
    },
}

impl TensorEncoding {
    /// Returns whether the value is stored densely.
    pub const fn is_dense(&self) -> bool {
        matches!(self, Self::Dense)
    }

    /// Returns whether the value is quantized.
    pub const fn is_quantized(&self) -> bool {
        !self.is_dense()
    }
}

impl Default for TensorEncoding {
    fn default() -> Self {
        Self::Dense
    }
}

/// A typed and shaped IR value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Value {
    /// Logical element type.
    pub dtype: DType,
    /// Logical shape; may contain symbolic dimensions in a `Program`.
    pub shape: Shape,
    /// Role the value plays.
    pub role: ValueRole,
    /// Physical storage encoding.
    pub encoding: TensorEncoding,
}

impl Value {
    /// Creates a densely stored value.
    pub fn new(dtype: DType, shape: Shape, role: ValueRole) -> Self {
        Self::with_encoding(dtype, shape, role, TensorEncoding::Dense)
    }

    /// Creates a value with an explicit storage encoding.
    pub fn with_encoding(
        dtype: DType,
        shape: Shape,
        role: ValueRole,
        encoding: TensorEncoding,
    ) -> Self {
        Self {
            dtype,
            shape,
            role,
            encoding,
        }
    }

    /// Returns whether the value is stored densely.
    pub fn is_dense(&self) -> bool {
        self.encoding.is_dense()
    }

    /// Returns the logical element count.
    pub fn element_count(&self) -> Result<u64, IrError> {
        self.shape.element_count()
    }

    /// Returns the storage size in bytes.
    ///
    /// For [`TensorEncoding::GroupwiseQuantized`] this is the packed payload
    /// size only; scale tensors are separate values. [`TensorEncoding::OpaqueQuantized`]
    /// has no known size and is rejected.
    pub fn storage_bytes(&self) -> Result<u64, IrError> {
        let elements = self.shape.element_count()?;
        match &self.encoding {
            TensorEncoding::Dense => elements
                .checked_mul(self.dtype.byte_width() as u64)
                .ok_or(IrError::ShapeOverflow),
            TensorEncoding::GroupwiseQuantized { format, group_size } => {
                if *group_size == 0 {
                    return Err(IrError::InvalidGroupSize);
                }
                let bits = elements
                    .checked_mul(u64::from(format.bits()))
                    .ok_or(IrError::ShapeOverflow)?;
                bits.checked_add(7)
                    .ok_or(IrError::ShapeOverflow)
                    .map(|rounded| rounded / 8)
            }
            TensorEncoding::OpaqueQuantized { .. } => Err(IrError::OpaqueEncodingSize),
        }
    }

    /// Alias for [`Self::storage_bytes`].
    pub fn byte_size(&self) -> Result<u64, IrError> {
        self.storage_bytes()
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
    use super::{QuantizationFormat, TensorEncoding, Value, ValueId, ValueRole, ValueRoleSet};
    use crate::{DType, IrError, Shape};
    use alloc::string::String;

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
        assert!(value.is_dense());
    }

    #[test]
    fn scalar_has_no_dimensions() {
        let scalar = Value::new(DType::U32, Shape::default(), ValueRole::Scalar);
        assert_eq!(scalar.byte_size().unwrap(), 4);
        assert!(scalar.shape.as_slice().is_empty());
        assert!(ValueRoleSet::ANY.contains(ValueRole::Output));
    }

    #[test]
    fn quantized_storage_uses_packed_payload_bytes() {
        let value = Value::with_encoding(
            DType::F32,
            Shape::consts(&[5]),
            ValueRole::Param,
            TensorEncoding::GroupwiseQuantized {
                format: QuantizationFormat::Int4,
                group_size: 4,
            },
        );
        assert_eq!(value.storage_bytes().unwrap(), 3);
        assert!(value.encoding.is_quantized());
    }

    #[test]
    fn opaque_encoding_has_no_known_size() {
        let value = Value::with_encoding(
            DType::F32,
            Shape::consts(&[4]),
            ValueRole::Param,
            TensorEncoding::OpaqueQuantized {
                scheme: String::from("unknown"),
            },
        );
        assert_eq!(value.storage_bytes(), Err(IrError::OpaqueEncodingSize));
    }
}
