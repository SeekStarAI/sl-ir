// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SeekStarAI

//! Opaque device handles and per-step bindings.

use crate::value::ValueId;
use alloc::vec::Vec;

/// Opaque buffer index interpreted by the backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct BufferId(pub u32);

/// Opaque queue index interpreted by the backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct QueueId(pub u32);

/// Opaque compiled-pipeline index interpreted by the backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct PipelineId(pub u32);

/// Binding of a persistent weight to a device buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ParamBinding {
    /// The parameter value.
    pub value: ValueId,
    /// The backing buffer.
    pub buffer: BufferId,
}

/// Per-step dynamic bindings.
///
/// Weights are bound once at pipeline-compile time and deliberately absent
/// here. All vectors are filled during engine build; `step` only updates
/// [`StepBinding::scalars`].
#[derive(Clone, Debug, Default)]
pub struct StepBinding {
    /// Per-step input buffers.
    pub inputs: Vec<(ValueId, BufferId)>,
    /// Per-step output buffers.
    pub outputs: Vec<(ValueId, BufferId)>,
    /// Per-step scalar values such as `seq_len` and `past_len`.
    pub scalars: Vec<(ValueId, u32)>,
    /// Persistent state buffers.
    pub states: Vec<(ValueId, BufferId)>,
    /// Scratchpad buffer, absent when the plan has no slots.
    pub scratchpad: Option<BufferId>,
}

impl StepBinding {
    /// Returns the buffer bound to an input value.
    pub fn input(&self, value: ValueId) -> Option<BufferId> {
        self.inputs
            .iter()
            .find(|(id, _)| *id == value)
            .map(|(_, buffer)| *buffer)
    }

    /// Updates the scalar value bound to a scalar input.
    pub fn set_scalar(&mut self, value: ValueId, new_value: u32) -> bool {
        if let Some(entry) = self.scalars.iter_mut().find(|(id, _)| *id == value) {
            entry.1 = new_value;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{BufferId, ParamBinding, StepBinding};
    use crate::value::ValueId;

    #[test]
    fn bindings_lookup_and_update_in_place() {
        let mut binding = StepBinding {
            inputs: alloc::vec![(ValueId(0), BufferId(1))],
            outputs: alloc::vec![(ValueId(1), BufferId(2))],
            scalars: alloc::vec![(ValueId(2), 7)],
            states: alloc::vec![(ValueId(3), BufferId(4))],
            scratchpad: Some(BufferId(5)),
        };

        assert_eq!(binding.input(ValueId(0)), Some(BufferId(1)));
        assert_eq!(binding.input(ValueId(9)), None);
        assert!(binding.set_scalar(ValueId(2), 9));
        assert!(!binding.set_scalar(ValueId(9), 9));
        assert_eq!(binding.scalars[0].1, 9);

        let param = ParamBinding {
            value: ValueId(4),
            buffer: BufferId(6),
        };
        assert_eq!(param.buffer, BufferId(6));
    }
}
