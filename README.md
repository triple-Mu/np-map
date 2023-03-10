# np-map

A simple tool for calculating object detection map .

# Install

```shell
pip install npmap -i https://pypi.org/simple
```

# Usage

```python
from npmap import ConfusionMatrix

matrix = ConfusionMatrix(num_classes=80)  # coco 80 classes

# cleanup first
matrix.clean()

for (pred, gt) in zip(preds, gts):  # for loop for every images
    # pred format x0, y0, x1, y1, scores, labels
    # gt format x0, y0, x1, y1, labels
    # if there is no gt or pred, please set pred=None or gt=None
    matrix.add(pred, gt)

# calculate map, map50, etc.
res = matrix.calculate()

# print map
print(res['map'])

# print map50
print(res['map50'])

# print mar
print(res['mean_recall'])

# print mp
print(res['mean_precision'])

# print per class results
print(res['detail'])

...  # so on
```
