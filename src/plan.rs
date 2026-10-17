// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SeekStarAI

//! Static scratchpad layout and its invariants.

use crate::command::Command;
use crate::error::IrError;
use crate::value::{Value, ValueId, ValueRole};
use alloc::vec;
use alloc::vec::Vec;

/// One scratchpad slot assigned to a `Temp` value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Slot {
    /// The value stored in this slot.
    pub value: ValueId,
    /// Byte offset inside the scratchpad.
    pub offset: u64,
    /// Logical payload size in bytes.
    pub bytes: u64,
}

/// Static scratchpad layout.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MemoryPlan {
    /// Total scratchpad size in bytes.
    pub scratchpad_bytes: u64,
    /// Slots sorted by value id.
    pub slots: Vec<Slot>,
}

impl MemoryPlan {
    /// Validates alignment, bounds, sizing, coverage and lifetime disjointness.
    ///
    /// `alignment` must be the backend's guaranteed buffer alignment.
    pub fn validate(
        &self,
        values: &[Value],
        commands: &[Command],
        alignment: u64,
    ) -> Result<(), IrError> {
        if alignment == 0 {
            return Err(IrError::PlanInvariant("alignment must be non-zero"));
        }

        let ranges = live_ranges(values.len(), commands);
        let mut seen = vec![false; values.len()];

        for slot in &self.slots {
            let index = slot.value.0 as usize;
            let value = values.get(index).ok_or(IrError::UnknownValue(slot.value))?;
            if value.role != ValueRole::Temp {
                return Err(IrError::PlanInvariant(
                    "only Temp values may occupy scratchpad slots",
                ));
            }
            if seen[index] {
                return Err(IrError::PlanInvariant(
                    "a value has more than one scratchpad slot",
                ));
            }
            seen[index] = true;

            if slot.offset % alignment != 0 {
                return Err(IrError::PlanInvariant("slot offset is not aligned"));
            }
            let end = slot
                .offset
                .checked_add(slot.bytes)
                .ok_or(IrError::PlanInvariant("slot range overflows"))?;
            if end > self.scratchpad_bytes {
                return Err(IrError::PlanInvariant("slot exceeds scratchpad"));
            }
            let expected = value.byte_size()?;
            if slot.bytes != expected {
                return Err(IrError::PlanInvariant(
                    "slot size does not match the value size",
                ));
            }
            if ranges[index].is_none() {
                return Err(IrError::PlanInvariant(
                    "Temp value is never referenced by a command",
                ));
            }
        }

        for (index, value) in values.iter().enumerate() {
            if value.role == ValueRole::Temp && !seen[index] {
                return Err(IrError::PlanInvariant(
                    "Temp value is missing from the memory plan",
                ));
            }
        }

        for (index, left) in self.slots.iter().enumerate() {
            for right in &self.slots[index + 1..] {
                if !ranges_overlap(left, right) {
                    continue;
                }
                let left_range = ranges[left.value.0 as usize]
                    .ok_or(IrError::PlanInvariant("slot value has no live range"))?;
                let right_range = ranges[right.value.0 as usize]
                    .ok_or(IrError::PlanInvariant("slot value has no live range"))?;
                if !(left_range.1 < right_range.0 || right_range.1 < left_range.0) {
                    return Err(IrError::PlanInvariant(
                        "overlapping slots have intersecting live ranges",
                    ));
                }
            }
        }

        Ok(())
    }
}

fn live_ranges(value_count: usize, commands: &[Command]) -> Vec<Option<(usize, usize)>> {
    let mut ranges = vec![None; value_count];
    for (index, command) in commands.iter().enumerate() {
        for value in command.inputs.iter().chain(command.outputs.iter()) {
            let slot = value.0 as usize;
            if slot >= value_count {
                continue;
            }
            match &mut ranges[slot] {
                None => ranges[slot] = Some((index, index)),
                Some((_, last)) => *last = index,
            }
        }
    }
    ranges
}

fn ranges_overlap(left: &Slot, right: &Slot) -> bool {
    let left_end = left.offset.saturating_add(left.bytes);
    let right_end = right.offset.saturating_add(right.bytes);
    left.offset < right_end && right.offset < left_end
}

#[cfg(test)]
mod tests {
    use super::{MemoryPlan, Slot};
    use crate::command::{Command, CommandId};
    use crate::error::IrError;
    use crate::op::Op;
    use crate::value::{Value, ValueId, ValueRole};
    use crate::{DType, Shape};
    use alloc::vec;

    fn temp() -> Value {
        Value::new(DType::F32, Shape::consts(&[4]), ValueRole::Temp)
    }

    fn command(inputs: Vec<ValueId>, outputs: Vec<ValueId>, deps: Vec<CommandId>) -> Command {
        Command {
            op: Op::Silu,
            inputs,
            outputs,
            deps,
            effect: crate::command::Effect::None,
        }
    }

    #[test]
    fn accepts_disjoint_reuse() {
        // v0: [0, 1], v2: [1, 2], v1: [2, 2]. v0 and v1 never coexist, so they
        // may share bytes; v2 overlaps both and must have its own slot.
        let values = vec![temp(), temp(), temp()];
        let commands = vec![
            command(vec![], vec![ValueId(0)], vec![]),
            command(vec![ValueId(0)], vec![ValueId(2)], vec![CommandId(0)]),
            command(vec![ValueId(2)], vec![ValueId(1)], vec![CommandId(1)]),
        ];
        let plan = MemoryPlan {
            scratchpad_bytes: 32,
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
                Slot {
                    value: ValueId(2),
                    offset: 16,
                    bytes: 16,
                },
            ],
        };

        assert_eq!(plan.validate(&values, &commands, 16), Ok(()));
    }

    #[test]
    fn rejects_misaligned_offset() {
        let values = vec![temp()];
        let commands = vec![command(vec![], vec![ValueId(0)], vec![])];
        let plan = MemoryPlan {
            scratchpad_bytes: 64,
            slots: vec![Slot {
                value: ValueId(0),
                offset: 8,
                bytes: 16,
            }],
        };

        assert_eq!(
            plan.validate(&values, &commands, 16).err(),
            Some(IrError::PlanInvariant("slot offset is not aligned"))
        );
    }

    #[test]
    fn rejects_missing_slot_and_wrong_role() {
        let values = vec![temp()];
        let commands = vec![command(vec![], vec![ValueId(0)], vec![])];
        assert_eq!(
            MemoryPlan::default()
                .validate(&values, &commands, 16)
                .err(),
            Some(IrError::PlanInvariant(
                "Temp value is missing from the memory plan"
            ))
        );

        let param = Value::new(DType::F32, Shape::consts(&[4]), ValueRole::Param);
        let plan = MemoryPlan {
            scratchpad_bytes: 16,
            slots: vec![Slot {
                value: ValueId(0),
                offset: 0,
                bytes: 16,
            }],
        };
        assert_eq!(
            plan.validate(&[param], &[], 16).err(),
            Some(IrError::PlanInvariant(
                "only Temp values may occupy scratchpad slots"
            ))
        );
    }
}
