# GPT-5.5 Secondary Review Evaluation

- Generated: 2026-06-15T16:23:48
- Total: 4
- Matched: 3
- Failed: 0
- Accuracy: 75.00%

## By Label

| Label | Total | Matched | Failed | Predicted | Avg Duration(s) |
|---|---:|---:|---:|---|---:|
| 正确 | 2 | 1 | 0 | {"clean": 1, "problematic": 1} | 29.82 |
| 误报 | 2 | 2 | 0 | {"problematic": 2} | 49.48 |

## Per Image

| Label | Image | Expected | Predicted | Match | Detections | FP | MD | Quality | Summary |
|---|---|---|---|---:|---:|---:|---:|---|---|
| 正确 | 00022.jpg | clean | clean | True | 1 | 0 | 0 | good | 检测框覆盖图中主要清晰可见的水泥搅拌/运输车辆，类别标为运输车可接受，位置基本准确；右侧仅有少量被遮挡车辆边缘，不作为清晰漏报目标。 |
| 正确 | 10_000060.jpg | clean | problematic | False | 3 | 1 | 1 | fair | 检测到的压路机和中部挖掘机位置与类别基本正确；第3个检测框将小型挖掘机误标为其他。另有一台中远处小型工程车辆较清晰但未被检测。 |
| 误报 | 20250304175558461.jpg | problematic | problematic | True | 1 | 0 | 2 | fair | 已有1个吊车检测框基本覆盖左侧橙色吊车，类别可接受；但施工区域中仍有至少两台清晰可见的吊装设备未被有效检测。 |
| 误报 | 20250307144503908.jpg | problematic | problematic | True | 5 | 3 | 2 | poor | 5个检测中有3个存在问题：第1个吊车框实际主要覆盖挖掘机，第4个挖掘机类别疑似应为铲车，第5个为绿色运输车重复框。另漏检了右侧清晰可见的大型吊车和黄色打桩设备。 |
