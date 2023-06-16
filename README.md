# Exploring Different Models to Achieve 99.4% accuracy with less than 8K Parameters under 15 Epochs using MNIST dataset.

# Model Varients

The model has four major versions:

1. Light Model
2. With Batch Normalization (BN)
3. With Batch Normalization (BN) and Dropout
4. With Batch Normalization (BN), Dropout, and Global Average Pooling (GAP)

## Light Model

**Target:** Building a simple model without any batch normalization, dropout, and global average pooling.

**Results:** 
- Parameters: 8.6K 
- Training Accuracy: 97.01%
- Test Accuracy: 98.02%

**Analysis:** Model might underperform compared to other versions due to the absence of BN, dropout and GAP also the numbers of parameters have dropped.

## With Batch Normalization (BN)

**Target:** Building a model with BN layers, added after each convolution layer to standardize feature maps. As Batch normalization is also to speed up learning and improve generalization.

**Results:** 
- Parameters: 8.6K 
- Training Accuracy: 98.76%
- Test Accuracy: 99.11%

**Analysis:** Batch normalization helped the model train faster and generalize better also the accuracy was improved.

## With Batch Normalization (BN) and Dropout

**Target:** Building a model with BN layers and DP layer, added after each convolution layer. Applying dropout for regularization and preventing overfitting.

**Results:** 
- Parameters: 8.6K 
- Training Accuracy: 97.85%
- Test Accuracy: 99.10%

**Analysis:** Dropout can help in managing overfitting by providing a form of regularization, the Training Accuracy dipped as expected and The test accuracy remained constant as per the previous model.

## With Batch Normalization (BN), Dropout, and GAP

**Target:** Create a model with batch normalization, dropout for regularization and global average pooling for better performance.

**Results:** 
- Parameters: 6.7K 
- Training Accuracy: 98.10%
- Test Accuracy: 98.16%

**Analysis:** This model was expected to perform best amongst all due to the combination of BN, dropout, and GAP. However, it might also be prone to overfitting due to its complexity, the results were not improved so let's try different LR values.

### LR Experimentation for Final Model 4

Final Model getting consistent 99.4% accuracy in last few epochs. With LR = 0.021

**Epoch: 12**
- Training Loss: 0.0547 
- Training Accuracy: 98.95%
- Test Loss: 0.0175
- Test Accuracy: 99.41%

**Epoch: 13**
- Training Loss: 0.0324 
- Training Accuracy: 98.98%
- Test Loss: 0.0162
- Test Accuracy: 99.43%

**Epoch: 14**
- Training Loss: 0.0126 
- Training Accuracy: 98.97%
- Test Loss: 0.0190
- Test Accuracy: 99.41%

**Epoch: 15**
- Training Loss: 0.0207 
- Training Accuracy: 98.98%
- Test Loss: 0.0173
- Test Accuracy: 99.44%

## Final Model Code

```python
self.model = nn.Sequential(
    nn.Conv2d(1, 8, 3, bias=False), # RF: 3
    nn.ReLU(),
    nn.BatchNorm2d(8),
    nn.Dropout(0.1),

    nn.Conv2d(8, 10, 3, bias=False), # RF: 5
    nn.ReLU(),
    nn.BatchNorm2d(10),
    nn.Dropout(0.1),

    nn.Conv2d(10, 12, 1, bias=False), # RF: 5
    nn.MaxPool2d(2, 2), # RF: 10

    nn.Conv2d(12, 14, 3, bias=False), # RF: 12
    nn.ReLU(),
    nn.BatchNorm2d(14),
    nn.Dropout(0.1),

    nn.Conv2d(14, 16, 3, bias=False), # RF: 14
    nn.ReLU(),
    nn.BatchNorm2d(16),
    nn.Dropout(0.1),

    nn.Conv2d(16, 8, 3, bias=False), # RF: 16
    nn.ReLU(),
    nn.BatchNorm2d(8),
    nn.Dropout(0.1),

    nn.Conv2d(8, 12, 3, padding=1, bias=False), # RF: 18
    nn.ReLU(),
    nn.BatchNorm2d(12),
    nn.Dropout(0.1),

    nn.AvgPool2d(6), # RF: 23
    nn.Conv2d(12, 10, 1, bias=False), # RF: 23
)

def forward(self, x):
    x = self.model(x)
    x = x.view(-1, 10)
    return F.log_softmax(x, dim=-1)
```

## Contributing

Contributions to this project are welcome! Please review the contribution guidelines prior to submitting a pull request.

