# sl-ir

StarLight 的计算图 IR 与设备命令 ABI 契约层。

本 crate 只包含**数据结构与不变量校验**，不包含 I/O、设备访问或任何优化算法。它是 `sl-graph`（编译器）、`sl-device`（设备契约）、`sl-model-adapters`（模型 lowering）、`sl-core`（运行时装配）与各 `backend-*` 之间共享的数据契约。

设计依据见 StarLight 主仓库的 [ARCH-0004](<../../docs/ARCH-0004: 计算图 IR 与设备命令 ABI 契约.md>)。

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

`Program` 负责表达“要算什么”，`CommandBuffer` 负责表达“在哪算、按什么顺序算、内存落在哪”。

## 当前能力

- `DType` / `DTypeSet`：`F32`、`F16`、`BF16`、`U32`、`I32`、`U8`、`I8`；
- `Dim` / `Shape` / `ShapeSpec`：符号维度、上界、`usize` 到具体形状的转换；
- `Value` / `ValueRole` / `ValueRoleSet`：值语义与角色；
- `TensorEncoding` / `QuantizationFormat`：`Dense`、`GroupwiseQuantized`、`OpaqueQuantized`；
  - 只有 `Param` 允许非稠密编码；激活、输入、输出、state、temp 必须是 `Dense`；
  - `Value::storage_bytes()` 对 Int4/Int8 返回打包载荷大小，scale 张量是独立 value；
- `ProgramBuilder`：为 Adapter 提供 `sym_dim` / `input` / `param` / `param_with_encoding` / `state` / `node` / `declare_input` / `declare_output` / `build`；
- `OpCode` / `Op` / `Signature`：算子词表、端口签名与节点签名校验；
- `Program` / `Node` / `Effect` / `StateDecl`：符号程序与状态副作用；
  - `Program::validate` 校验端口、dtype、角色、秩、符号维、state 形状与存储编码；
- `Command` / `CommandBuffer`：物理命令流；
- `MemoryPlan` / `Slot`：静态 scratchpad 布局与不变量校验；
- `DeviceCapabilities`：backend 能力声明；
- `ParamBinding` / `StepBinding`：编译期与每步绑定。

尚未提供：能力协商 pass、内存着色算法、`Program → CommandBuffer` 的具体化实现。这些属于 `sl-graph`；量化算子的执行属于 backend，当前 `sl-core` 会在构建期拒绝非稠密参数。

## 版本

`IR_VERSION` 当前为 **2**。

- v1：初始 ABI；
- v2：新增 `U8`/`I8`、`Value::encoding`、`TensorEncoding`，并把 state 形状与非稠密编码校验下沉到 `sl-ir`。

## 设计约束

- **零依赖**。不引入 `serde`、`thiserror`、`anyhow` 或日志库。
- **`no_std` 可行**。默认开启 `std` feature；关闭后只依赖 `alloc`。
- **数据与算法分离**。本 crate 可以校验不变量，但不得实现优化、调度或着色。
- **`IR_VERSION` 独立于 crate semver**。任何影响 ABI 语义的改动都必须提升 `IR_VERSION`。

## 使用

```rust
use sl_ir::{DType, Dim, Op, ProgramBuilder, Shape};

let mut builder = ProgramBuilder::new();
let seq = builder.sym_dim("seq", 8);

let input = builder.input(
    DType::F32,
    Shape::new(vec![Dim::Const(1), Dim::Sym(seq), Dim::Const(8)]),
);
let output = builder.output(
    DType::F32,
    Shape::new(vec![Dim::Const(1), Dim::Sym(seq), Dim::Const(8)]),
);

builder
    .node(Op::Silu, vec![input], vec![output], sl_ir::Effect::None)
    .declare_input(input)
    .declare_output(output);

let program = builder.build()?;
# Ok::<(), sl_ir::IrError>(())
```

量化参数：

```rust
use sl_ir::{DType, ProgramBuilder, QuantizationFormat, Shape, TensorEncoding};

let mut builder = ProgramBuilder::new();
let weight = builder.param_with_encoding(
    DType::F32,
    Shape::consts(&[4096, 4096]),
    TensorEncoding::GroupwiseQuantized {
        format: QuantizationFormat::Int4,
        group_size: 128,
    },
);
assert_eq!(builder.build().unwrap().values[weight.0 as usize].storage_bytes().unwrap(), 4096 * 4096 * 4 / 8);
```

## 测试

```sh
cargo test -p sl-ir
cargo clippy -p sl-ir --all-targets -- -D warnings
cargo check -p sl-ir --no-default-features
```

当前 32 个单元测试，覆盖签名校验、state 形状、编码与打包字节数、builder 与内存计划不变量。

## 许可

MIT，见 [LICENSE](LICENSE)。
