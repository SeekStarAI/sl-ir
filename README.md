# sl-ir

StarLight 的计算图 IR 与设备命令 ABI 契约层。

本 crate 只包含**数据结构与不变量校验**，不包含 I/O、设备访问或任何优化算法。它是 `sl-graph`（编译器）、`sl-device`（设备契约）、`sl-model-adapters`（模型 lowering）、`sl-core`（运行时装配）与各 `backend-*` 之间共享的数据契约。

设计依据见 StarLight 主仓库的 [ARCH-0004：计算图 IR 与设备命令 ABI 契约](../../docs/ARCH-0004:%20计算图%20IR%20与设备命令%20ABI%20契约.md)。

## 两层 IR

```text
Program（符号）
  形状可含 Dim::Sym，由模型 Adapter 构造
        │
        │  sl-graph::compile(program, shape_spec, capabilities)
        ▼
CommandBuffer（物理）
  形状全部具体化，含调度顺序与静态内存布局，由 backend 消费
```

`Program` 负责表达"要算什么"，`CommandBuffer` 负责表达"在哪算、按什么顺序算、内存落在哪"。

## 当前能力（M0）

- `DType` / `DTypeSet`：元素类型与集合。
- `Dim` / `Shape` / `ShapeSpec`：符号维度、上界与具体化。
- `Value` / `ValueRole` / `ValueRoleSet`：值语义与角色。
- `OpCode` / `Op` / `Signature`：算子词表与端口签名。
- `Program` / `Node` / `Effect` / `StateDecl`：符号程序与状态副作用。
- `Command` / `CommandBuffer`：物理命令流。
- `MemoryPlan` / `Slot`：静态 scratchpad 布局与不变量校验。
- `DeviceCapabilities`：backend 能力声明。
- `ParamBinding` / `StepBinding`：编译期与每步绑定。

尚未提供：算子签名的完整校验、能力协商 pass、内存着色算法、`Program → CommandBuffer` 的具体化实现。这些属于 `sl-graph`。

## 设计约束

- **零依赖**。不引入 `serde`、`thiserror`、`anyhow` 或日志库。
- **`no_std` 可行**。默认开启 `std` feature；关闭后只依赖 `alloc`。
- **数据与算法分离**。本 crate 可以校验不变量，但不得实现优化、调度或着色。
- **`IR_VERSION` 独立于 crate semver**。任何影响 ABI 语义的改动都必须提升 `IR_VERSION`。

## 使用

```rust
use sl_ir::{DType, Dim, Op, Program, Shape, Value, ValueRole, ValueId};

let program = Program {
    sym_dims: vec![],
    values: vec![
        Value::new(DType::F32, Shape::consts(&[1, 8]), ValueRole::Input),
        Value::new(DType::F32, Shape::consts(&[1, 8]), ValueRole::Temp),
    ],
    nodes: vec![sl_ir::Node {
        op: Op::Silu,
        inputs: vec![ValueId(0)],
        outputs: vec![ValueId(1)],
        effect: sl_ir::Effect::None,
    }],
    inputs: vec![ValueId(0)],
    outputs: vec![ValueId(1)],
    states: vec![],
};

program.validate()?;
# Ok::<(), sl_ir::IrError>(())
```

## 测试

```sh
cargo test -p sl-ir
cargo clippy -p sl-ir --all-targets -- -D warnings
cargo test -p sl-ir --no-default-features   # no_std + alloc 路径
```

## 许可

MIT，见 [LICENSE](LICENSE)。
