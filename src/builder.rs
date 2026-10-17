// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SeekStarAI

//! Ergonomic construction of [`Program`]s by model adapters.
//!
//! The builder allocates value ids, records symbolic dimensions, nodes, inputs,
//! outputs and states, and validates the finished program. It deliberately
//! contains no lowering logic: adapters decide the graph, the builder only
//! enforces structural invariants.

use crate::command::{Effect, Node, Program, StateDecl, StateKind};
use crate::dtype::DType;
use crate::error::IrError;
use crate::op::Op;
use crate::shape::{Shape, SymDimDecl, SymId};
use crate::value::{TensorEncoding, Value, ValueId, ValueRole};
use alloc::vec::Vec;

/// Builder for [`Program`].
#[derive(Clone, Debug, Default)]
pub struct ProgramBuilder {
    sym_dims: Vec<SymDimDecl>,
    values: Vec<Value>,
    nodes: Vec<Node>,
    inputs: Vec<ValueId>,
    outputs: Vec<ValueId>,
    states: Vec<StateDecl>,
}

impl ProgramBuilder {
    /// Creates an empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Declares a symbolic dimension and returns its id.
    pub fn sym_dim(&mut self, name: &'static str, max: u32) -> SymId {
        let id = SymId(
            u32::try_from(self.sym_dims.len()).expect("symbolic dimension count exceeds u32::MAX"),
        );
        self.sym_dims.push(SymDimDecl { id, name, max });
        id
    }

    /// Appends a densely stored value.
    pub fn value(&mut self, dtype: DType, shape: Shape, role: ValueRole) -> ValueId {
        self.value_with_encoding(dtype, shape, role, TensorEncoding::Dense)
    }

    /// Appends a value with an explicit storage encoding.
    pub fn value_with_encoding(
        &mut self,
        dtype: DType,
        shape: Shape,
        role: ValueRole,
        encoding: TensorEncoding,
    ) -> ValueId {
        let id = ValueId(u32::try_from(self.values.len()).expect("value count exceeds u32::MAX"));
        self.values
            .push(Value::with_encoding(dtype, shape, role, encoding));
        id
    }

    /// Appends a per-step input value.
    pub fn input(&mut self, dtype: DType, shape: Shape) -> ValueId {
        self.value(dtype, shape, ValueRole::Input)
    }

    /// Appends a per-step scalar value.
    pub fn scalar(&mut self, dtype: DType, shape: Shape) -> ValueId {
        self.value(dtype, shape, ValueRole::Scalar)
    }

    /// Appends a per-step output value.
    pub fn output(&mut self, dtype: DType, shape: Shape) -> ValueId {
        self.value(dtype, shape, ValueRole::Output)
    }

    /// Appends a persistent parameter with dense storage.
    pub fn param(&mut self, dtype: DType, shape: Shape) -> ValueId {
        self.value(dtype, shape, ValueRole::Param)
    }

    /// Appends a persistent parameter with an explicit storage encoding.
    pub fn param_with_encoding(
        &mut self,
        dtype: DType,
        shape: Shape,
        encoding: TensorEncoding,
    ) -> ValueId {
        self.value_with_encoding(dtype, shape, ValueRole::Param, encoding)
    }

    /// Appends a scratchpad temporary.
    pub fn temp(&mut self, dtype: DType, shape: Shape) -> ValueId {
        self.value(dtype, shape, ValueRole::Temp)
    }

    /// Appends a persistent state and its declaration.
    pub fn state(&mut self, dtype: DType, shape: Shape, kind: StateKind) -> ValueId {
        let id = self.value(dtype, shape, ValueRole::State);
        self.states.push(StateDecl { value: id, kind });
        id
    }

    /// Appends a node.
    pub fn node(
        &mut self,
        op: Op,
        inputs: Vec<ValueId>,
        outputs: Vec<ValueId>,
        effect: Effect,
    ) -> &mut Self {
        self.nodes.push(Node {
            op,
            inputs,
            outputs,
            effect,
        });
        self
    }

    /// Declares a value as a program input.
    pub fn declare_input(&mut self, value: ValueId) -> &mut Self {
        self.inputs.push(value);
        self
    }

    /// Declares a value as a program output.
    pub fn declare_output(&mut self, value: ValueId) -> &mut Self {
        self.outputs.push(value);
        self
    }

    /// Builds and validates the program.
    pub fn build(self) -> Result<Program, IrError> {
        let program = Program {
            sym_dims: self.sym_dims,
            values: self.values,
            nodes: self.nodes,
            inputs: self.inputs,
            outputs: self.outputs,
            states: self.states,
        };
        program.validate()?;
        Ok(program)
    }
}

#[cfg(test)]
mod tests {
    use super::ProgramBuilder;
    use crate::command::Effect;
    use crate::dtype::DType;
    use crate::error::IrError;
    use crate::op::Op;
    use crate::shape::Shape;
    use crate::value::{QuantizationFormat, TensorEncoding, ValueRole};

    #[test]
    fn builds_a_valid_program() {
        let mut builder = ProgramBuilder::new();
        let seq = builder.sym_dim("seq", 8);

        let token_ids = builder.input(
            DType::U32,
            Shape::new(vec![crate::Dim::Const(1), crate::Dim::Sym(seq)]),
        );
        let table = builder.param(DType::F32, Shape::consts(&[8, 4]));
        let rows = builder.temp(
            DType::F32,
            Shape::new(vec![
                crate::Dim::Const(1),
                crate::Dim::Sym(seq),
                crate::Dim::Const(4),
            ]),
        );
        let out = builder.output(
            DType::F32,
            Shape::new(vec![
                crate::Dim::Const(1),
                crate::Dim::Sym(seq),
                crate::Dim::Const(4),
            ]),
        );

        builder
            .node(Op::Gather, vec![table, token_ids], vec![rows], Effect::None)
            .node(Op::Silu, vec![rows], vec![out], Effect::None)
            .declare_input(token_ids)
            .declare_output(out);

        let program = builder
            .build()
            .expect("builder must produce a valid program");
        assert_eq!(program.values.len(), 4);
        assert_eq!(program.nodes.len(), 2);
        assert_eq!(program.sym_dims[0].name, "seq");
    }

    #[test]
    fn builds_quantized_parameters() {
        let mut builder = ProgramBuilder::new();
        let id = builder.param_with_encoding(
            DType::F32,
            Shape::consts(&[8, 4]),
            TensorEncoding::GroupwiseQuantized {
                format: QuantizationFormat::Int4,
                group_size: 4,
            },
        );
        let program = builder.build().expect("quantized parameter is allowed");
        assert!(program.values[id.0 as usize].encoding.is_quantized());
    }

    #[test]
    fn rejects_non_dense_temporaries() {
        let mut builder = ProgramBuilder::new();
        builder.value_with_encoding(
            DType::F32,
            Shape::consts(&[4]),
            ValueRole::Temp,
            TensorEncoding::GroupwiseQuantized {
                format: QuantizationFormat::Int8,
                group_size: 4,
            },
        );
        assert_eq!(
            builder.build().err(),
            Some(IrError::NonDenseValue {
                value: crate::ValueId(0),
                role: ValueRole::Temp,
            })
        );
    }
}
