// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SeekStarAI

//! Symbolic programs, physical command streams and their effects.

use crate::dtype::DType;
use crate::error::IrError;
use crate::op::{Op, PortSpec};
use crate::plan::MemoryPlan;
use crate::shape::{Dim, SymDimDecl};
use crate::value::{Value, ValueId, ValueRole};
use alloc::vec::Vec;

/// Index of a command inside a [`CommandBuffer`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct CommandId(pub u32);

/// Read/write effect of a node or command on persistent state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Effect {
    /// No persistent state is touched.
    None,
    /// The command reads the given state value.
    ReadState(ValueId),
    /// The command writes the given state value.
    WriteState(ValueId),
}

/// Semantic layout of a persistent state value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StateKind {
    /// Key/value cache of a decoder-only model.
    KvCache {
        /// Batch size.
        batch: u32,
        /// Number of layers.
        layers: u32,
        /// Number of key/value heads.
        kv_heads: u32,
        /// Attention head width.
        head_dim: u32,
        /// Maximum sequence length.
        max_seq: Dim,
        /// Element type.
        dtype: DType,
    },
}

/// Declaration of a persistent state value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StateDecl {
    /// The state value.
    pub value: ValueId,
    /// Semantic layout.
    pub kind: StateKind,
}

/// One node of a symbolic [`Program`].
#[derive(Clone, Debug)]
pub struct Node {
    /// Operator and its attributes.
    pub op: Op,
    /// Input values.
    pub inputs: Vec<ValueId>,
    /// Output values.
    pub outputs: Vec<ValueId>,
    /// Effect on persistent state.
    pub effect: Effect,
}

/// Symbolic program produced by a model adapter.
///
/// Shapes may contain symbolic dimensions; `sl-graph` specializes them into a
/// [`CommandBuffer`] before execution.
#[derive(Clone, Debug)]
pub struct Program {
    /// Symbolic dimension declarations.
    pub sym_dims: Vec<SymDimDecl>,
    /// Value table; ids index this vector directly.
    pub values: Vec<Value>,
    /// Computation nodes in program order.
    pub nodes: Vec<Node>,
    /// Program inputs.
    pub inputs: Vec<ValueId>,
    /// Program outputs.
    pub outputs: Vec<ValueId>,
    /// Persistent state declarations.
    pub states: Vec<StateDecl>,
}

impl Program {
    /// Looks up a value by id.
    pub fn value(&self, id: ValueId) -> Result<&Value, IrError> {
        self.values
            .get(id.0 as usize)
            .ok_or(IrError::UnknownValue(id))
    }

    /// Validates structure, roles, symbolic declarations and state effects.
    pub fn validate(&self) -> Result<(), IrError> {
        for (index, decl) in self.sym_dims.iter().enumerate() {
            if self.sym_dims[..index]
                .iter()
                .any(|other| other.id == decl.id)
            {
                return Err(IrError::DuplicateSymbolicDim(decl.id));
            }
        }

        for id in &self.inputs {
            let value = self.value(*id)?;
            if value.role != ValueRole::Input {
                return Err(IrError::InvalidRole {
                    value: *id,
                    expected: ValueRole::Input,
                    actual: value.role,
                });
            }
            self.check_symbols(value)?;
        }
        for id in &self.outputs {
            let value = self.value(*id)?;
            if value.role != ValueRole::Output {
                return Err(IrError::InvalidRole {
                    value: *id,
                    expected: ValueRole::Output,
                    actual: value.role,
                });
            }
            self.check_symbols(value)?;
        }

        for (index, node) in self.nodes.iter().enumerate() {
            for id in node.inputs.iter().chain(node.outputs.iter()) {
                self.check_symbols(self.value(*id)?)?;
            }
            match node.effect {
                Effect::None => {}
                Effect::ReadState(id) | Effect::WriteState(id) => {
                    let value = self.value(id)?;
                    if value.role != ValueRole::State {
                        return Err(IrError::InvalidEffect { node: index });
                    }
                    if !self.states.iter().any(|state| state.value == id) {
                        return Err(IrError::StateNotDeclared(id));
                    }
                }
            }
            self.check_signature(index, node)?;
        }

        for (index, state) in self.states.iter().enumerate() {
            let value = self.value(state.value)?;
            if value.role != ValueRole::State {
                return Err(IrError::InvalidRole {
                    value: state.value,
                    expected: ValueRole::State,
                    actual: value.role,
                });
            }
            if self.states[..index]
                .iter()
                .any(|other| other.value == state.value)
            {
                return Err(IrError::DuplicateState(state.value));
            }
        }

        Ok(())
    }

    fn check_symbols(&self, value: &Value) -> Result<(), IrError> {
        for dim in value.shape.as_slice() {
            if let Dim::Sym(sym) = dim {
                if !self.sym_dims.iter().any(|decl| decl.id == *sym) {
                    return Err(IrError::UndeclaredSymbolicDim(*sym));
                }
            }
        }
        Ok(())
    }

    fn check_signature(&self, index: usize, node: &Node) -> Result<(), IrError> {
        let signature = node.op.code().signature();
        if node.inputs.len() != signature.inputs.len() {
            return Err(IrError::ArityMismatch {
                node: index,
                expected: signature.inputs.len(),
                actual: node.inputs.len(),
            });
        }
        if node.outputs.len() != signature.outputs.len() {
            return Err(IrError::ArityMismatch {
                node: index,
                expected: signature.outputs.len(),
                actual: node.outputs.len(),
            });
        }
        for (port_index, (port, id)) in signature.inputs.iter().zip(node.inputs.iter()).enumerate()
        {
            self.check_port(index, port_index, port, *id)?;
        }
        for (port_index, (port, id)) in signature
            .outputs
            .iter()
            .zip(node.outputs.iter())
            .enumerate()
        {
            self.check_port(index, port_index, port, *id)?;
        }
        Ok(())
    }

    fn check_port(
        &self,
        node: usize,
        port_index: usize,
        port: &PortSpec,
        id: ValueId,
    ) -> Result<(), IrError> {
        let value = self.value(id)?;
        if !port.dtypes.contains(value.dtype) {
            return Err(IrError::DTypeNotAllowed {
                node,
                dtype: value.dtype,
            });
        }
        if !port.roles.contains(value.role) {
            return Err(IrError::RoleNotAllowed {
                node,
                value: id,
                role: value.role,
            });
        }
        let rank = value.shape.rank();
        if rank < port.min_rank as usize || rank > port.max_rank as usize {
            return Err(IrError::RankMismatch {
                node,
                port: port_index,
                rank,
                min: port.min_rank,
                max: port.max_rank,
            });
        }
        if !port.allow_symbolic && value.shape.has_symbolic() {
            return Err(IrError::SymbolicNotAllowed { node });
        }
        Ok(())
    }
}

/// One command of a [`CommandBuffer`].
#[derive(Clone, Debug)]
pub struct Command {
    /// Operator and its attributes.
    pub op: Op,
    /// Input values.
    pub inputs: Vec<ValueId>,
    /// Output values; may be empty for effect-only operators.
    pub outputs: Vec<ValueId>,
    /// Preceding commands this command depends on.
    pub deps: Vec<CommandId>,
    /// Effect on persistent state.
    pub effect: Effect,
}

/// Physical command stream consumed by a backend.
#[derive(Clone, Debug)]
pub struct CommandBuffer {
    /// IR version this buffer was compiled against.
    pub ir_version: u32,
    /// Value table with fully concrete shapes.
    pub values: Vec<Value>,
    /// Commands in execution order.
    pub commands: Vec<Command>,
    /// Static scratchpad layout.
    pub plan: MemoryPlan,
    /// Buffer inputs.
    pub inputs: Vec<ValueId>,
    /// Buffer outputs.
    pub outputs: Vec<ValueId>,
    /// Persistent state declarations.
    pub states: Vec<StateDecl>,
}

impl CommandBuffer {
    /// Looks up a value by id.
    pub fn value(&self, id: ValueId) -> Result<&Value, IrError> {
        self.values
            .get(id.0 as usize)
            .ok_or(IrError::UnknownValue(id))
    }

    /// Validates version, shapes, roles, ordering, effects and the memory plan.
    pub fn validate(&self, alignment: u64) -> Result<(), IrError> {
        if self.ir_version != crate::IR_VERSION {
            return Err(IrError::IrVersionMismatch {
                found: self.ir_version,
                expected: crate::IR_VERSION,
            });
        }

        for (index, value) in self.values.iter().enumerate() {
            if value.shape.has_symbolic() {
                return Err(IrError::SymbolicDimInCommandBuffer(ValueId(index as u32)));
            }
        }

        for (index, state) in self.states.iter().enumerate() {
            let value = self.value(state.value)?;
            if value.role != ValueRole::State {
                return Err(IrError::InvalidRole {
                    value: state.value,
                    expected: ValueRole::State,
                    actual: value.role,
                });
            }
            let StateKind::KvCache { max_seq, .. } = state.kind;
            if max_seq.is_symbolic() {
                return Err(IrError::SymbolicDimInCommandBuffer(state.value));
            }
            if self.states[..index]
                .iter()
                .any(|other| other.value == state.value)
            {
                return Err(IrError::DuplicateState(state.value));
            }
        }

        for id in &self.inputs {
            let value = self.value(*id)?;
            if value.role != ValueRole::Input {
                return Err(IrError::InvalidInputOutput { value: *id });
            }
        }
        for id in &self.outputs {
            let value = self.value(*id)?;
            if value.role != ValueRole::Output {
                return Err(IrError::InvalidInputOutput { value: *id });
            }
        }

        for (index, command) in self.commands.iter().enumerate() {
            for dependency in &command.deps {
                if dependency.0 as usize >= index {
                    return Err(IrError::CommandOutOfOrder {
                        command: index,
                        dependency: dependency.0,
                    });
                }
            }
            for id in command.inputs.iter().chain(command.outputs.iter()) {
                self.value(*id)?;
            }
            match command.effect {
                Effect::None => {}
                Effect::ReadState(id) | Effect::WriteState(id) => {
                    let value = self.value(id)?;
                    if value.role != ValueRole::State {
                        return Err(IrError::InvalidEffect { node: index });
                    }
                    if !self.states.iter().any(|state| state.value == id) {
                        return Err(IrError::StateNotDeclared(id));
                    }
                }
            }
        }

        self.plan.validate(&self.values, &self.commands, alignment)
    }
}
