# Overview

Various notes for later.

# Log

**Thu Apr 9 23:54:59 2020 -0400**

* added ability to save/load tokenization file to save time
* added three jobs to cluster testing latest transformer with 100,150,200 sized batchs
* pulled existing tf checkpoint to local machine to experiment with
* set vocab length to 50k to match paper


## Sample Training 1

* local computer
* 50k dataset
* small transformer

```
Epoch 1 Batch 0 Loss 5.3179 Accuracy 0.0000
Epoch 1 Batch 50 Loss 5.1245 Accuracy 0.0007
Epoch 1 Batch 100 Loss 5.0493 Accuracy 0.0121
Epoch 1 Batch 150 Loss 4.9640 Accuracy 0.0179
Epoch 1 Batch 200 Loss 4.9049 Accuracy 0.0213
Epoch 1 Batch 250 Loss 4.8280 Accuracy 0.0248
Epoch 1 Batch 300 Loss 4.7325 Accuracy 0.0288
Epoch 1 Batch 350 Loss 4.6301 Accuracy 0.0339
Epoch 1 Batch 400 Loss 4.5354 Accuracy 0.0387
Epoch 1 Batch 450 Loss 4.4469 Accuracy 0.0427
Epoch 1 Batch 500 Loss 4.3617 Accuracy 0.0468
Epoch 1 Loss 4.3543 Accuracy 0.0472
Time taken for 1 epoch: 429.6082499027252 secs

Epoch 2 Batch 0 Loss 3.6000 Accuracy 0.0861
Epoch 2 Batch 50 Loss 3.5589 Accuracy 0.0938
Epoch 2 Batch 100 Loss 3.4928 Accuracy 0.0976
Epoch 2 Batch 150 Loss 3.4432 Accuracy 0.1009
Epoch 2 Batch 200 Loss 3.3948 Accuracy 0.1043
Epoch 2 Batch 250 Loss 3.3360 Accuracy 0.1073
Epoch 2 Batch 300 Loss 3.2924 Accuracy 0.1102
Epoch 2 Batch 350 Loss 3.2560 Accuracy 0.1132
Epoch 2 Batch 400 Loss 3.2203 Accuracy 0.1157
Epoch 2 Batch 450 Loss 3.1881 Accuracy 0.1182
Epoch 2 Batch 500 Loss 3.1576 Accuracy 0.1206
Epoch 2 Loss 3.1546 Accuracy 0.1208
Time taken for 1 epoch: 508.1735608577728 secs

Epoch 3 Batch 0 Loss 2.6967 Accuracy 0.1466
Epoch 3 Batch 50 Loss 2.8358 Accuracy 0.1469
Epoch 3 Batch 100 Loss 2.8165 Accuracy 0.1470
Epoch 3 Batch 150 Loss 2.8147 Accuracy 0.1480
Epoch 3 Batch 200 Loss 2.8061 Accuracy 0.1490
Epoch 3 Batch 250 Loss 2.7812 Accuracy 0.1499
Epoch 3 Batch 300 Loss 2.7709 Accuracy 0.1512
Epoch 3 Batch 350 Loss 2.7631 Accuracy 0.1524
Epoch 3 Batch 400 Loss 2.7522 Accuracy 0.1536
Epoch 3 Batch 450 Loss 2.7423 Accuracy 0.1545
Epoch 3 Batch 500 Loss 2.7298 Accuracy 0.1554
Epoch 3 Loss 2.7307 Accuracy 0.1555
Time taken for 1 epoch: 375.77764987945557 secs

Epoch 4 Batch 0 Loss 2.3700 Accuracy 0.1530
Epoch 4 Batch 50 Loss 2.5652 Accuracy 0.1665
Epoch 4 Batch 100 Loss 2.5757 Accuracy 0.1674
Epoch 4 Batch 150 Loss 2.5758 Accuracy 0.1688
Epoch 4 Batch 200 Loss 2.5688 Accuracy 0.1695
Epoch 4 Batch 250 Loss 2.5600 Accuracy 0.1701
Epoch 4 Batch 300 Loss 2.5548 Accuracy 0.1708
Epoch 4 Batch 350 Loss 2.5466 Accuracy 0.1713
Epoch 4 Batch 400 Loss 2.5400 Accuracy 0.1718
Epoch 4 Batch 450 Loss 2.5300 Accuracy 0.1723
Epoch 4 Batch 500 Loss 2.5223 Accuracy 0.1729
Epoch 4 Loss 2.5218 Accuracy 0.1730
Time taken for 1 epoch: 377.37636494636536 secs

Epoch 5 Batch 0 Loss 2.4920 Accuracy 0.1854
Epoch 5 Batch 50 Loss 2.3952 Accuracy 0.1818
Epoch 5 Batch 100 Loss 2.3878 Accuracy 0.1826
Epoch 5 Batch 150 Loss 2.3787 Accuracy 0.1832
Epoch 5 Batch 200 Loss 2.3773 Accuracy 0.1835
Epoch 5 Batch 250 Loss 2.3736 Accuracy 0.1842
Epoch 5 Batch 300 Loss 2.3718 Accuracy 0.1842
Epoch 5 Batch 350 Loss 2.3696 Accuracy 0.1845
Epoch 5 Batch 400 Loss 2.3648 Accuracy 0.1846
Epoch 5 Batch 450 Loss 2.3623 Accuracy 0.1850
Epoch 5 Batch 500 Loss 2.3608 Accuracy 0.1855
Saving checkpoint for epoch 5 at checkpoints/train/ckpt-1
Epoch 5 Loss 2.3611 Accuracy 0.1855
Time taken for 1 epoch: 369.0688579082489 secs

Epoch 6 Batch 0 Loss 2.3648 Accuracy 0.1935
Epoch 6 Batch 50 Loss 2.2233 Accuracy 0.1959
Epoch 6 Batch 100 Loss 2.2535 Accuracy 0.1973
Epoch 6 Batch 150 Loss 2.2392 Accuracy 0.1965
Epoch 6 Batch 200 Loss 2.2297 Accuracy 0.1961
Epoch 6 Batch 250 Loss 2.2260 Accuracy 0.1962
Epoch 6 Batch 300 Loss 2.2241 Accuracy 0.1959
Epoch 6 Batch 350 Loss 2.2222 Accuracy 0.1962
Epoch 6 Batch 400 Loss 2.2209 Accuracy 0.1966
Epoch 6 Batch 450 Loss 2.2219 Accuracy 0.1968
Epoch 6 Batch 500 Loss 2.2155 Accuracy 0.1972
Epoch 6 Loss 2.2157 Accuracy 0.1972
Time taken for 1 epoch: 362.1720447540283 secs

Epoch 7 Batch 0 Loss 2.0631 Accuracy 0.2047
Epoch 7 Batch 50 Loss 2.0691 Accuracy 0.2105
Epoch 7 Batch 100 Loss 2.0718 Accuracy 0.2096
Epoch 7 Batch 150 Loss 2.0799 Accuracy 0.2105
Epoch 7 Batch 200 Loss 2.0769 Accuracy 0.2110
Epoch 7 Batch 250 Loss 2.0680 Accuracy 0.2104
Epoch 7 Batch 300 Loss 2.0710 Accuracy 0.2108
Epoch 7 Batch 350 Loss 2.0657 Accuracy 0.2109
Epoch 7 Batch 400 Loss 2.0678 Accuracy 0.2112
Epoch 7 Batch 450 Loss 2.0652 Accuracy 0.2113
Epoch 7 Batch 500 Loss 2.0684 Accuracy 0.2119
Epoch 7 Loss 2.0678 Accuracy 0.2119
Time taken for 1 epoch: 374.4852149486542 secs

Epoch 8 Batch 0 Loss 1.8588 Accuracy 0.2323
Epoch 8 Batch 50 Loss 1.9147 Accuracy 0.2254
Epoch 8 Batch 100 Loss 1.9154 Accuracy 0.2253
Epoch 8 Batch 150 Loss 1.9243 Accuracy 0.2260
Epoch 8 Batch 200 Loss 1.9210 Accuracy 0.2253
Epoch 8 Batch 250 Loss 1.9204 Accuracy 0.2253
Epoch 8 Batch 300 Loss 1.9211 Accuracy 0.2251
Epoch 8 Batch 350 Loss 1.9264 Accuracy 0.2253
Epoch 8 Batch 400 Loss 1.9262 Accuracy 0.2254
Epoch 8 Batch 450 Loss 1.9265 Accuracy 0.2256
Epoch 8 Batch 500 Loss 1.9284 Accuracy 0.2257
Epoch 8 Loss 1.9300 Accuracy 0.2257
Time taken for 1 epoch: 396.97506499290466 secs

Epoch 9 Batch 0 Loss 1.7301 Accuracy 0.2228
Epoch 9 Batch 50 Loss 1.7721 Accuracy 0.2395
Epoch 9 Batch 100 Loss 1.7858 Accuracy 0.2407
Epoch 9 Batch 150 Loss 1.7957 Accuracy 0.2407
Epoch 9 Batch 200 Loss 1.7956 Accuracy 0.2405
Epoch 9 Batch 250 Loss 1.7902 Accuracy 0.2405
Epoch 9 Batch 300 Loss 1.7901 Accuracy 0.2402
Epoch 9 Batch 350 Loss 1.7957 Accuracy 0.2403
Epoch 9 Batch 400 Loss 1.7942 Accuracy 0.2404
Epoch 9 Batch 450 Loss 1.7961 Accuracy 0.2404
Epoch 9 Batch 500 Loss 1.7937 Accuracy 0.2403
Epoch 9 Loss 1.7928 Accuracy 0.2402
Time taken for 1 epoch: 399.38680696487427 secs

Epoch 10 Batch 0 Loss 1.7763 Accuracy 0.2817
Epoch 10 Batch 50 Loss 1.6325 Accuracy 0.2585
Epoch 10 Batch 100 Loss 1.6342 Accuracy 0.2598
Epoch 10 Batch 150 Loss 1.6435 Accuracy 0.2585
Epoch 10 Batch 200 Loss 1.6485 Accuracy 0.2580
Epoch 10 Batch 250 Loss 1.6516 Accuracy 0.2576
Epoch 10 Batch 300 Loss 1.6498 Accuracy 0.2566
Epoch 10 Batch 350 Loss 1.6576 Accuracy 0.2563
Epoch 10 Batch 400 Loss 1.6611 Accuracy 0.2563
Epoch 10 Batch 450 Loss 1.6625 Accuracy 0.2556
Epoch 10 Batch 500 Loss 1.6624 Accuracy 0.2554
Saving checkpoint for epoch 10 at checkpoints/train/ckpt-2
Epoch 10 Loss 1.6622 Accuracy 0.2554
Time taken for 1 epoch: 402.072527885437 secs
```

## Sample Training 2

* local computer
* 50k dataset
* large transformer

```
Epoch 1 Batch 0 Loss 4.8953 Accuracy 0.0000
Epoch 1 Batch 50 Loss 4.9216 Accuracy 0.0144
Epoch 1 Batch 100 Loss 4.7744 Accuracy 0.0219
Epoch 1 Batch 150 Loss 4.6650 Accuracy 0.0265
Epoch 1 Batch 200 Loss 4.5316 Accuracy 0.0338
Epoch 1 Batch 250 Loss 4.3834 Accuracy 0.0405
Epoch 1 Batch 300 Loss 4.2625 Accuracy 0.0461
Epoch 1 Batch 350 Loss 4.1607 Accuracy 0.0510
Epoch 1 Batch 400 Loss 4.0795 Accuracy 0.0562
Epoch 1 Batch 450 Loss 4.0067 Accuracy 0.0611
Epoch 1 Batch 500 Loss 3.9354 Accuracy 0.0659
Epoch 1 Loss 3.9305 Accuracy 0.0664
Time taken for 1 epoch: 2523.215889930725 secs

Epoch 2 Batch 0 Loss 3.0965 Accuracy 0.1106
Epoch 2 Batch 50 Loss 3.2388 Accuracy 0.1188
Epoch 2 Batch 100 Loss 3.1933 Accuracy 0.1223
Epoch 2 Batch 150 Loss 3.1549 Accuracy 0.1261
Epoch 2 Batch 200 Loss 3.1120 Accuracy 0.1293
Epoch 2 Batch 250 Loss 3.0824 Accuracy 0.1321
Epoch 2 Batch 300 Loss 3.0424 Accuracy 0.1342
Epoch 2 Batch 350 Loss 3.0186 Accuracy 0.1361
Epoch 2 Batch 400 Loss 2.9925 Accuracy 0.1379
Epoch 2 Batch 450 Loss 2.9649 Accuracy 0.1396
Epoch 2 Batch 500 Loss 2.9421 Accuracy 0.1415
Epoch 2 Loss 2.9403 Accuracy 0.1417
Time taken for 1 epoch: 2244.2350430488586 secs
```