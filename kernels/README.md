# kernels

这里预留给以后的 CUDA / Triton 实现。当前能跑的 decode kernel 仍在 `src/logitfill/packed/kernel.cu`，由 `kernel_loader.py` 在第一次 packed decode 时编译，失败则退回 dense reference。这一步没有搬它。

后续要补的内容：

- 支持的 CUDA / Triton 版本
- 编译命令
- 与 PyTorch 的接口约定
- 已知限制
