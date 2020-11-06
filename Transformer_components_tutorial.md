# Transformer Components Tutorial

---

Last updated: 11/05/2020

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

