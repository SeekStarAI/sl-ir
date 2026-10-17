// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SeekStarAI

//! Dimensions, shapes, symbolic bounds and shape constraints.

use crate::dtype::DType;
use crate::error::IrError;
use alloc::vec::Vec;

/// Identifier of a symbolic dimension.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SymId(pub u32);

/// A single tensor dimension.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Dim {
    /// A concrete extent.
    Const(u32),
    /// A symbolic extent resolved at compile time.
    Sym(SymId),
}

impl Dim {
    /// Returns whether the dimension is symbolic.
    pub const fn is_symbolic(self) -> bool {
        matches!(self, Self::Sym(_))
    }

    /// Returns the concrete extent, if any.
    pub const fn as_const(self) -> Option<u32> {
        match self {
            Self::Const(value) => Some(value),
            Self::Sym(_) => None,
        }
    }
}

/// A tensor shape in logical order.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Shape(Vec<Dim>);

impl Shape {
    /// Creates a shape from dimensions.
    pub fn new(dims: Vec<Dim>) -> Self {
        Self(dims)
    }

    /// Creates a fully concrete shape.
    pub fn consts(dims: &[u32]) -> Self {
        Self(dims.iter().copied().map(Dim::Const).collect())
    }

    /// Returns the dimensions.
    pub fn as_slice(&self) -> &[Dim] {
        &self.0
    }

    /// Returns the number of dimensions.
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// Returns whether any dimension is symbolic.
    pub fn has_symbolic(&self) -> bool {
        self.0.iter().any(|dim| dim.is_symbolic())
    }

    /// Returns the logical element count.
    ///
    /// Fails for symbolic shapes; resolve them with [`ShapeSpec`] first.
    pub fn element_count(&self) -> Result<u64, IrError> {
        self.0.iter().try_fold(1u64, |count, dim| match dim {
            Dim::Const(value) => count.checked_mul(u64::from(*value)).ok_or(IrError::ShapeOverflow),
            Dim::Sym(id) => Err(IrError::UnresolvedSymbolicDim(*id)),
        })
    }

    /// Returns the storage size in bytes.
    pub fn byte_size(&self, dtype: DType) -> Result<u64, IrError> {
        self.element_count()?
            .checked_mul(dtype.byte_width() as u64)
            .ok_or(IrError::ShapeOverflow)
    }

    /// Returns the concrete extents, or `None` if any dimension is symbolic.
    pub fn to_consts(&self) -> Option<Vec<u32>> {
        self.0.iter().map(|dim| dim.as_const()).collect()
    }
}

/// Declaration of a symbolic dimension and its compile-time upper bound.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SymDimDecl {
    /// Symbolic dimension identifier.
    pub id: SymId,
    /// Human-readable name used in diagnostics.
    pub name: &'static str,
    /// Largest extent accepted by compilation.
    pub max: u32,
}

/// One entry of a [`ShapeSpec`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShapeSpecEntry {
    /// Symbolic dimension identifier.
    pub id: SymId,
    /// Largest extent accepted by compilation.
    pub max: u32,
    /// Extent used to specialize the program.
    pub value: u32,
}

/// Compile-time mapping from symbolic dimensions to concrete extents.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ShapeSpec {
    entries: Vec<ShapeSpecEntry>,
}

impl ShapeSpec {
    /// Creates a specification. Duplicate ids keep the first entry.
    pub fn new(mut entries: Vec<ShapeSpecEntry>) -> Self {
        entries.sort_by_key(|entry| entry.id);
        entries.dedup_by_key(|entry| entry.id);
        Self { entries }
    }

    /// Returns the entries.
    pub fn entries(&self) -> &[ShapeSpecEntry] {
        &self.entries
    }

    /// Looks up the extent assigned to a symbolic dimension.
    pub fn get(&self, id: SymId) -> Option<u32> {
        self.entries
            .binary_search_by_key(&id, |entry| entry.id)
            .ok()
            .map(|index| self.entries[index].value)
    }

    /// Replaces symbolic dimensions with concrete extents.
    pub fn resolve(&self, shape: &Shape) -> Result<Shape, IrError> {
        let mut dims = Vec::with_capacity(shape.rank());
        for dim in shape.as_slice() {
            match *dim {
                Dim::Const(value) => dims.push(Dim::Const(value)),
                Dim::Sym(id) => {
                    let index = self
                        .entries
                        .binary_search_by_key(&id, |entry| entry.id)
                        .map_err(|_| IrError::UndeclaredSymbolicDim(id))?;
                    let entry = self.entries[index];
                    if entry.value > entry.max {
                        return Err(IrError::SymbolicDimExceedsMax {
                            dim: id,
                            max: entry.max,
                            requested: entry.value,
                        });
                    }
                    dims.push(Dim::Const(entry.value));
                }
            }
        }
        Ok(Shape::new(dims))
    }
}

/// Backend-specific shape constraints checked during compilation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ShapeConstraints {
    /// Required multiple for `head_dim`; `0` disables the check.
    pub head_dim_multiple_of: u32,
    /// Required multiple for the reduction dimension of matmuls; `0` disables.
    pub matmul_k_multiple_of: u32,
    /// Maximum number of attention heads; `0` disables the check.
    pub max_heads: u32,
}

#[cfg(test)]
mod tests {
    use super::{Dim, Shape, ShapeSpec, ShapeSpecEntry, SymId};
    use crate::error::IrError;
    use alloc::vec;

    #[test]
    fn concrete_shapes_report_element_and_byte_counts() {
        let shape = Shape::consts(&[2, 3]);
        assert_eq!(shape.element_count().unwrap(), 6);
        assert_eq!(shape.byte_size(crate::DType::F32).unwrap(), 24);
        assert!(!shape.has_symbolic());
        assert_eq!(shape.to_consts().unwrap(), vec![2, 3]);
    }

    #[test]
    fn zero_dimensional_shape_is_a_scalar() {
        let scalar = Shape::default();
        assert_eq!(scalar.rank(), 0);
        assert_eq!(scalar.element_count().unwrap(), 1);
    }

    #[test]
    fn empty_extent_produces_empty_tensor() {
        let empty = Shape::consts(&[0, 4]);
        assert_eq!(empty.element_count().unwrap(), 0);
        assert_eq!(empty.byte_size(crate::DType::F32).unwrap(), 0);
    }

    #[test]
    fn resolve_rejects_missing_and_oversized_symbols() {
        let id = SymId(3);
        let shape = Shape::new(vec![Dim::Sym(id)]);
        assert_eq!(
            ShapeSpec::default().resolve(&shape).err(),
            Some(IrError::UndeclaredSymbolicDim(id))
        );

        let spec = ShapeSpec::new(vec![ShapeSpecEntry {
            id,
            max: 4,
            value: 8,
        }]);
        assert_eq!(
            spec.resolve(&shape).err(),
            Some(IrError::SymbolicDimExceedsMax {
                dim: id,
                max: 4,
                requested: 8,
            })
        );
    }

    #[test]
    fn duplicate_spec_entries_keep_the_first_value() {
        let id = SymId(1);
        let spec = ShapeSpec::new(vec![
            ShapeSpecEntry {
                id,
                max: 10,
                value: 2,
            },
            ShapeSpecEntry {
                id,
                max: 10,
                value: 7,
            },
        ]);
        assert_eq!(spec.get(id), Some(2));
    }
}
