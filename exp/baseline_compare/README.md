# baseline 对比

同一段冻结 token 上，比较 LogitFill、KIVI、KVTuner、PM-KVQ 的逐步 ΔNLL。MixKVQ 用自己的引擎，主环境的 transformers 4.49 导入不了它。

```bash
python exp/baseline_compare/run.py --config exp/baseline_compare/config.yaml
```
