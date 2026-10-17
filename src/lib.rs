// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SeekStarAI

//! Backend-agnostic graph IR and device command ABI for StarLight.
//!
//! This crate holds data contracts only: no I/O, no device access and no
//! optimization algorithms. The two IR layers are [`Program`] (symbolic, built
//! by model adapters) and [`CommandBuffer`] (physical, produced by `sl-graph`
//! and consumed by backends).
//!
//! The design rationale lives in `docs/ARCH-0004` of the StarLight workspace.

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(unsafe_code)]

extern crate alloc;

pub mod binding;
pub mod caps;
pub mod command;
pub mod dtype;
pub mod error;
pub mod op;
pub mod plan;
pub mod shape;
pub mod value;

/// Version of the IR data model and command ABI.
///
/// Bumped on any change to operator semantics, command layout, memory-plan
/// invariants or binding structures. Backends must declare the version they
/// were built against.
pub const IR_VERSION: u32 = 1;

pub use binding::{BufferId, ParamBinding, PipelineId, QueueId, StepBinding};
pub use caps::{BackendTag, DeviceCapabilities};
pub use command::{
    Command, CommandBuffer, CommandId, Effect, Node, Program, StateDecl, StateKind,
};
pub use dtype::{DType, DTypeSet};
pub use error::IrError;
pub use op::{Op, OpCode, OpSet, PortSpec, RopeVariant, Signature};
pub use plan::{MemoryPlan, Slot};
pub use shape::{Dim, Shape, ShapeConstraints, ShapeSpec, ShapeSpecEntry, SymDimDecl, SymId};
pub use value::{Value, ValueId, ValueRole, ValueRoleSet};

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    fn scalar() -> Value {
        Value::new(DType::U32, Shape::default(), ValueRole::Scalar)
    }

    fn f32_value(shape: &[u32], role: ValueRole) -> Value {
        Value::new(DType::F32, Shape::consts(shape), role)
    }

    #[test]
    fn dtype_sets_are_ordered_and_queryable() {
        let floats = DTypeSet::FLOATS;
        assert!(floats.contains(DType::F32));
        assert!(floats.contains(DType::BF16));
        assert!(!floats.contains(DType::U32));
        assert_eq!(floats.to_vec(), vec![DType::F32, DType::F16, DType::BF16]);
    }

    #[test]
    fn shape_resolution_substitutes_and_checks_bounds() {
        let seq = SymId(0);
        let shape = Shape::new(vec![Dim::Const(1), Dim::Sym(seq)]);
        let spec = ShapeSpec::new(vec![ShapeSpecEntry {
            id: seq,
            max: 2048,
            value: 128,
        }]);

        assert_eq!(spec.resolve(&shape).unwrap().to_consts().unwrap(), vec![1, 128]);
        assert_eq!(
            shape.element_count().err(),
            Some(IrError::UnresolvedSymbolicDim(seq))
        );

        let too_large = ShapeSpec::new(vec![ShapeSpecEntry {
            id: seq,
            max: 16,
            value: 32,
        }]);
        assert_eq!(
            too_large.resolve(&shape).err(),
            Some(IrError::SymbolicDimExceedsMax {
                dim: seq,
                max: 16,
                requested: 32,
            })
        );
    }

    #[test]
    fn program_rejects_undeclared_state_effect() {
        let mut program = Program {
            sym_dims: Vec::new(),
            values: vec![f32_value(&[1, 2], ValueRole::Input)],
            nodes: vec![Node {
                op: Op::KvAppend { layer: 0 },
                inputs: vec![ValueId(0)],
                outputs: Vec::new(),
                effect: Effect::WriteState(ValueId(0)),
            }],
            inputs: vec![ValueId(0)],
            outputs: Vec::new(),
            states: Vec::new(),
        };
        assert_eq!(program.validate().err(), Some(IrError::InvalidEffect { node: 0 }));

        program.values[0] = f32_value(&[1, 2], ValueRole::State);
        program.inputs.clear();
        assert_eq!(
            program.validate().err(),
            Some(IrError::StateNotDeclared(ValueId(0)))
        );
    }

    #[test]
    fn program_validates_operator_signatures() {
        let mut program = Program {
            sym_dims: Vec::new(),
            values: vec![
                f32_value(&[2, 4], ValueRole::Temp),
                f32_value(&[3, 4], ValueRole::Temp),
                f32_value(&[2, 3], ValueRole::Temp),
            ],
            nodes: vec![Node {
                op: Op::MatMulNT,
                inputs: vec![ValueId(0)],
                outputs: vec![ValueId(2)],
                effect: Effect::None,
            }],
            inputs: Vec::new(),
            outputs: Vec::new(),
            states: Vec::new(),
        };

        assert_eq!(
            program.validate().err(),
            Some(IrError::ArityMismatch {
                node: 0,
                expected: 2,
                actual: 1,
            })
        );

        program.nodes[0].inputs = vec![ValueId(0), ValueId(1)];
        assert_eq!(program.validate(), Ok(()));

        program.values[1] = Value::new(DType::F32, Shape::default(), ValueRole::Temp);
        assert_eq!(
            program.validate().err(),
            Some(IrError::RankMismatch {
                node: 0,
                port: 1,
                rank: 0,
                min: 2,
                max: 4,
            })
        );

        program.values[1] = Value::new(DType::U32, Shape::consts(&[3, 4]), ValueRole::Temp);
        assert_eq!(
            program.validate().err(),
            Some(IrError::DTypeNotAllowed {
                node: 0,
                dtype: DType::U32,
            })
        );

        program.values[1] = Value::new(DType::F32, Shape::consts(&[3, 4]), ValueRole::State);
        assert_eq!(
            program.validate().err(),
            Some(IrError::RoleNotAllowed {
                node: 0,
                value: ValueId(1),
                role: ValueRole::State,
            })
        );
    }

    #[test]
    fn memory_plan_rejects_overlapping_live_ranges() {
        let values = vec![
            f32_value(&[4], ValueRole::Temp),
            f32_value(&[4], ValueRole::Temp),
        ];
        let commands = vec![
            Command {
                op: Op::Silu,
                inputs: vec![ValueId(0)],
                outputs: vec![ValueId(1)],
                deps: Vec::new(),
                effect: Effect::None,
            },
            Command {
                op: Op::Scale { factor: 1.0 },
                inputs: vec![ValueId(0)],
                outputs: Vec::new(),
                deps: vec![CommandId(0)],
                effect: Effect::None,
            },
        ];
        let plan = MemoryPlan {
            scratchpad_bytes: 64,
            slots: vec![
                Slot {
                    value: ValueId(0),
                    offset: 0,
                    bytes: 16,
                },
                Slot {
                    value: ValueId(1),
                    offset: 0,
                    bytes: 16,
                },
            ],
        };

        assert_eq!(
            plan.validate(&values, &commands, 64).err(),
            Some(IrError::PlanInvariant(
                "overlapping slots have intersecting live ranges"
            ))
        );
    }

    #[test]
    fn command_buffer_rejects_symbolic_dims() {
        let buffer = CommandBuffer {
            ir_version: IR_VERSION,
            values: vec![Value::new(
                DType::F32,
                Shape::new(vec![Dim::Sym(SymId(0))]),
                ValueRole::Temp,
            )],
            commands: Vec::new(),
            plan: MemoryPlan::default(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            states: Vec::new(),
        };

        assert_eq!(
            buffer.validate(64).err(),
            Some(IrError::SymbolicDimInCommandBuffer(ValueId(0)))
        );
    }

    #[test]
    fn scalar_values_have_one_element() {
        assert_eq!(scalar().byte_size().unwrap(), 4);
    }
}
