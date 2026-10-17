# sl-ir Development Guide

## Responsibility

`sl-ir` owns the data contracts shared by the StarLight graph compiler, runtime, model
adapters and device backends. It defines *what* data looks like, never *how* it is
computed, scheduled or optimized.

Dependency direction:

```text
sl-model ──> sl-model-adapters ──> Program ──┐
                                             ├──> sl-ir
sl-graph (compiler) ─────────────────────────┤
sl-device (traits) ──────────────────────────┤
backend-* ───────────────────────────────────┘
```

`sl-ir` depends on nothing. It must never depend on `sl-model`, `sl-graph`, `sl-device`,
any backend or any third-party crate.

## Module Map

| Module | Contents |
| --- | --- |
| `dtype` | `DType`, `DTypeSet` |
| `shape` | `Dim`, `Shape`, `ShapeSpec`, `SymDimDecl`, `ShapeConstraints` |
| `value` | `Value`, `ValueId`, `ValueRole`, `ValueRoleSet` |
| `op` | `OpCode`, `Op`, `Signature`, `PortSpec`, `OpSet`, `RopeVariant` |
| `command` | `Program`, `Node`, `Effect`, `StateDecl`, `Command`, `CommandBuffer` |
| `plan` | `MemoryPlan`, `Slot`, plan invariants |
| `caps` | `DeviceCapabilities`, `BackendTag` |
| `binding` | `BufferId`, `ParamBinding`, `StepBinding` |
| `error` | `IrError` |

## Hard Rules

1. **No dependencies.** `[dependencies]` must stay empty. Dev-dependencies may be added
   only if they do not enter the public API.
2. **No algorithms.** Validation is allowed; optimization, scheduling, liveness analysis
   beyond invariant checking, and graph rewriting are not.
3. **No `std` requirement.** The crate must compile with `--no-default-features`.
   Use `alloc` types and `core` traits only.
4. **No `unsafe`.** `#![deny(unsafe_code)]` is intentional.
5. **No `String` in errors.** `IrError` is `Copy`; formatting happens at the boundary.
6. **`ValueId` indexes `values` directly.** Constructors and validators must preserve
   this invariant.
7. **Discriminants are ABI.** `OpCode` values are never reused; removed operators keep
   their number as a tombstone.

## Invariants Enforced Today

- Program inputs have role `Input`, outputs have role `Output`.
- Every symbolic dimension used by a value is declared in `Program::sym_dims`.
- State effects target values with role `State` and an entry in `Program::states`.
- `CommandBuffer` contains no symbolic dimensions, including KV `max_seq`.
- Command dependencies always reference earlier commands.
- Memory plan slots are aligned, in bounds, correctly sized, unique and cover every
  `Temp` value.
- Values sharing scratchpad bytes have disjoint live ranges.

## Validation Commands

Run from the StarLight workspace root:

```sh
cargo test -p sl-ir
cargo clippy -p sl-ir --all-targets -- -D warnings
cargo test -p sl-ir --no-default-features
```

## Change Process

Any change to `OpCode` variants, `Signature`, `MemoryPlan` invariants or binding
structures changes the ABI. Such a change requires:

1. an update to `IR_VERSION`;
2. a matching update to ARCH-0004 in the StarLight workspace;
3. a backend capability update in every backend crate that declares an `ir_version`.

Additive Rust APIs that do not alter the data model may ship without an `IR_VERSION`
bump, but still need tests and documentation.
