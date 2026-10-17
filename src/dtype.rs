// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SeekStarAI

//! Element types shared by the IR, capabilities and backends.

use alloc::vec::Vec;
use core::fmt;

/// Element type of an IR value.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum DType {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
    U32 = 3,
    I32 = 4,
    /// Unsigned 8-bit storage; used for packed quantized payloads.
    U8 = 5,
    /// Signed 8-bit storage.
    I8 = 6,
}

impl DType {
    /// Number of defined element types.
    pub const COUNT: usize = 7;

    /// Returns the stable bit index used by [`DTypeSet`].
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Returns the storage width in bits.
    pub const fn bit_width(self) -> usize {
        match self {
            Self::F32 | Self::U32 | Self::I32 => 32,
            Self::F16 | Self::BF16 => 16,
            Self::U8 | Self::I8 => 8,
        }
    }

    /// Returns the storage width in bytes.
    pub const fn byte_width(self) -> usize {
        self.bit_width() / 8
    }

    /// Returns whether the type is a floating-point type.
    pub const fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F16 | Self::BF16)
    }

    /// Returns every defined element type in declaration order.
    pub const fn all() -> [Self; Self::COUNT] {
        [
            Self::F32,
            Self::F16,
            Self::BF16,
            Self::U32,
            Self::I32,
            Self::U8,
            Self::I8,
        ]
    }
}

impl fmt::Display for DType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::U32 => "U32",
            Self::I32 => "I32",
            Self::U8 => "U8",
            Self::I8 => "I8",
        })
    }
}

/// Compact set of element types stored as a bit mask.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DTypeSet(u16);

impl DTypeSet {
    /// Empty set.
    pub const EMPTY: Self = Self(0);

    /// Set containing only `F32`.
    pub const ONLY_F32: Self = Self(1 << 0);

    /// Set containing only `U32`.
    pub const ONLY_U32: Self = Self(1 << 3);

    /// Set containing only `U8`.
    pub const ONLY_U8: Self = Self(1 << 5);

    /// All floating-point types.
    pub const FLOATS: Self = Self((1 << 0) | (1 << 1) | (1 << 2));

    /// All integer types.
    pub const INTEGERS: Self = Self((1 << 3) | (1 << 4) | (1 << 5) | (1 << 6));

    /// Every defined type.
    pub const ANY: Self = Self((1u16 << DType::COUNT) - 1);

    /// Creates a set containing a single type.
    pub const fn of(dtype: DType) -> Self {
        Self(1u16 << (dtype.index() as u16))
    }

    /// Returns the union of two sets.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Returns whether the set contains `dtype`.
    pub const fn contains(self, dtype: DType) -> bool {
        self.0 & (1u16 << (dtype.index() as u16)) != 0
    }

    /// Returns whether the set is empty.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Returns the contained types in declaration order.
    pub fn to_vec(self) -> Vec<DType> {
        DType::all()
            .into_iter()
            .filter(|dtype| self.contains(*dtype))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{DType, DTypeSet};
    use alloc::vec;

    #[test]
    fn masks_report_membership() {
        assert!(DTypeSet::ANY.contains(DType::I32));
        assert!(DTypeSet::ANY.contains(DType::U8));
        assert!(!DTypeSet::INTEGERS.contains(DType::F32));
        assert!(DTypeSet::INTEGERS.contains(DType::U8));
        assert!(DTypeSet::EMPTY.is_empty());
        assert_eq!(DTypeSet::ANY.to_vec().len(), DType::COUNT);
        assert_eq!(
            DTypeSet::ONLY_F32.union(DTypeSet::ONLY_U32).to_vec(),
            vec![DType::F32, DType::U32]
        );
    }
}
