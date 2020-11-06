# Transformer Components Tutorial

---

Last updated: 11/05/2020

#### References

1. The original Transformer paper [arXiv](https://arxiv.org/abs/1706.03762)

2. Implementation of Transformer by TensorFlow [link](https://www.tensorflow.org/tutorials/text/transformer)

3. Implementation of Transformer by Trax were learn from the course [Natural Language Processing with Attention](https://www.coursera.org/learn/attention-models-in-nlp) on *Coursera*

## Contents

1. Attention


## Dot-Product Attention


## Causal Attention

Causal attention only look at the previous words. The gist of implementation is to use a mask to neglect the words in the future.

The mask (M) can be created by setting the values 0 at previous locations, while setting values -inf at future locations.

And the attention model calculates

**softmax(QK' + M)**

(K' is the transpose of K)

Since future locations were added by -inf, the softmax of that location is close to zero. Therefore the attention will not be placed at the future positions.

#### Tips

1. Thesoftmax calculation can be done by using `scipy.special.logsumexp` to avoid underflow

``` python3
exponent = scipy.special.logsumexp(QK', axis = -1, keepdims = True)

res = np.exp(dots - exponent)
```

2. The mask boolean matrix can be created by `np.tril`

```python3
mask_boolMatrix = np.tril(np.ones((1, mask_size, mask_size), dtype = np.bool_))
```

3. The mask can be added to QK' by using `np.where` and `np.full_like`

``` python3
QK' = np.where(mask_boolMatrix, QK', np.full_like(QK', -1e9))
```

## Tensor Dimensions

Below summaries the shapes of the tensors at each stage:

| Stage | Shape |
| ----------- | ----------- |
| Input (Q, K, V) | (batch_size, sentence_length, embedding_dim) |
| Linear layer | (batch_size, sentence_length, n_heads * d_head) |
| Transpose | (batch_size, n_heads, sentence_length, d_head) |
| After attention | (batch_size, n_heads, sentence_length, d_head) |
| Transpose | (batch_size, sentence_length, n_heads * d_head) |
| Linear layer | (batch_size, sentence_length, embedding_dim) |
