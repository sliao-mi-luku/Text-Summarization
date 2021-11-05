# Text Summarization with Transformer-based Language Models

Implementing Transformer-based models to draft a summary from an article

[![text-summarization-cartoon.jpg](https://i.postimg.cc/7Y98jH5p/text-summarization-cartoon.jpg)](https://postimg.cc/bsZBtjHT)
<p align="center">
    Text summarization
</p>

## Project Summary
After training the model *from scratch* on the cnn_dailymail dataset for 500,000 epochs, the model achieved the accuracy of **56.98%** on the test set.

To see how the model performs, below is an article drawn from the test dataset:

#### Original Article:
The NFL have fined the Atlanta Falcons and stripped them of a draft pick following the team's use of fake crowd noise at home games. In a statement released on Monday, the league announced that the Falcons have been fined $350,000 (\xc2\xa3237,000)\xc2\xa0and will forfeit their fifth-round selection in the 2016 draft. If the Falcons have multiple picks in that round, the highest selection will be forfeited. Team president Rich McKay has also been suspended from the league's Competition Committee beginning April 1. Atlanta Falcons have been fined \xc2\xa3237,000 for their use of fake crowd noise at the Georgia Dome . Owner Arthur Blank acknowledged the team's wrongdoing and described the incident as embarrassing . The NFL noted throughout the 2013 season and into the 2014 season the Falcons violated league rules that state 'at no point during the game can artificial crowd noise or amplified crowd noise be played in the stadium.' The league also said Roddy White, the team's former director of event marketing, was directly responsible for the violation and would have been suspended without pay for the first eight weeks of the 2015 regular season had he still been with the club. The Falcons fired him. The league determined that Falcons ownership and senior executives, including McKay, were unaware of the use of an audio file with artificial crowd noise. But as the senior club executive overseeing game operations, McKay bears some responsibility for ensuring that team employees comply with league rules. McKay can petition Commissioner Roger Goodell for reinstatement to the committee no sooner than June 30. The Falcons played fake crowd noise during the 2013 and 2014 seasons . Falcons president Rick McKay has been suspended from his position on NFL Competition Committee . Falcons owner Arthur Blank said in early February that he had seen enough of the NFL's investigation to acknowledge wrongdoing by his club. 'It's not really a fine line,' Blank said. 'I think what we've done in 2013 and 2014 was wrong. Anything that affects the competitive balance and fairness on the field, we're opposed to, as a league, as a club and as an owner. It's obviously embarrassing but beyond embarrassing it doesn't represent our culture and what we're about.' The Falcons say 101 of 103 games have been sellouts since Blank bought the team in 2002. Actual turnouts declined during losing seasons the last two years. Atlanta ranked 10th among the 32 NFL teams with its average home attendance of 72,130 in 2014. Construction is underway for a new $1.4 billion stadium that will replace the Georgia Dome in 2017. The new stadium will have a similar seating capacity.

#### Actual Summary:
- Atlanta Falcons played fake crowd noise during 2013 and 2014 season .
- Falcons fined $350,000 (\xc2\xa3237,000) and been stripped of a 2016 draft pick .
- President Rich McKay has been suspended from Competition Committee .

#### Model Predictions:
- Atlanta Falcons will forfeit their fifth-round selection in the 2016 draft .
- Falcons have been fined £280,000 for their use of fake crowd noise .
- The Falcons have been fined £280,000 for their use of fake crowd noise .
- The Falcons have been fined £280,000 for their use of fake crowd noise .

The model learns the keys in this story: "forfeit their fifth-round selection", "fined", and "use of fake crowd noise"!


## Data

I use 2 different datasets: `multi_news` and `cnn_dailymail`

**multi_news**

The multi_news dataset is 245 MB. It contains 44,972 training data.

The articles and summaries can be loaded by calling keys = ('documents', 'summary')

**cnn_dailymail**

The cnn_dailymail dataset is ? MB. It contains ? training data.

The articles and summaries can be loaded by calling keys = ('article', 'highlights')

## Model

I implemented a Transformer model from scratch for this task.

[![transformer-full-architecture.png](https://i.postimg.cc/FFV4gps8/transformer-full-architecture.png)](https://postimg.cc/s1xq3pW4)
<p align="center">
    Transformer architecture. Image  adapted from the original paper "Attention is all you need"
</p>

## Code

Below are the codes for building a Transformer:

#### Multi-Head Attention (MHA)
```python
def MHA(d_model, n_heads, mode = 'train'):

    """
    Multi-Head Attention

    Inputs
            d_model: <int> dimension of input embedding
            n_heads: <int> number of heads of each layer of mha
            mode: <str> 'train', 'eval', or 'predict'

    Output
    """

    # make sure d_model / n_head = d_k is an integer
    assert d_model % n_heads == 0
    d_k = d_model // n_heads

    def mha_input_func(x):
        """
        The function to reshape the input tensor x
        """
        # the batch_size
        batch_size= x.shape[0]
        # the input sequence length
        seq_len = x.shape[1]
        # reshape: (batch_size, seq_len, d_model) --> (batch_size, seq_len, n_heads, d_k)
        x = fastnp.reshape(x, (batch_size, seq_len, n_heads, d_k))
        # transpose: (batch_size, seq_len, n_heads, d_k) --> (batch_size, d_k, seq_len, n_heads)
        x = fastnp.transpose(x, (0, 2, 1, 3))
        # reshape: (batch_size, d_k, seq_len, n_heads) --> (batch_size * n_heads, seq_len, d_k)
        x = fastnp.reshape(x, (batch_size * n_heads, seq_len, d_k))
        return x

    def mha_attention_func(Q, K, V):
        """
        Calculate the attention, given (query, key, value) = (Q, K, V)
        """
        # the size of the mask
        mask_size = Q.shape[1]
        # create the mask (M)
        M = fastnp.tril(fastnp.ones((1, mask_size, mask_size), dtype = fastnp.bool_))
        # calculate the scaled dot-product
        scaled_dot_product = fastnp.matmul(Q, fastnp.swapaxes(K, 1, 2)) / fastnp.sqrt(d_k)
        # apply mask
        if M is not None:
            scaled_dot_product = fastnp.where(M, scaled_dot_product, fastnp.full_like(scaled_dot_product, -1e9))
        # softmax
        softmax_scaled_dot_product = fastnp.exp(scaled_dot_product - trax.fastmath.logsumexp(scaled_dot_product, axis = -1, keepdims = True))
        # matmul with V
        attention = fastnp.matmul(softmax_scaled_dot_product, V)
        return attention

    def mha_output_func(x):
        """
        The function to reshape the output tensor x of the attention
        x.shape = (batch_size * n_heads, seqlen, d_head)
        """
        # the sequence length
        seq_len = x.shape[1]
        # reshape: (batch_size * n_heads, seqlen, d_head) --> (batch_size, n_heads, seq_len, d_k)
        x = fastnp.reshape(x, (-1, n_heads, seq_len, d_k))
        # transpose: (batch_size, n_heads, seq_len, d_k) --> (batch_size, seq_len, n_heads, d_k)
        x = fastnp.transpose(x, (0, 2, 1, 3))
        # reshape: (batch_size, seq_len, n_heads, d_k) --> (batch_size, seq_len, n_heads * d_k)
        x = fastnp.reshape(x, (-1, seq_len, n_heads * d_k))
        return x

    ## Convert functions to layers
    mha_input_layer = tl.Fn("mha_input", mha_input_func, n_out = 1)
    mha_attention_layer = tl.Fn("mha_attention", mha_attention_func, n_out = 1)
    mha_output_layer = tl.Fn("mha_output", mha_output_func, n_out = 1)

    ## MHA layer
    MHA_layer = tl.Serial(tl.Branch([tl.Dense(d_model), mha_input_layer],
                                    [tl.Dense(d_model), mha_input_layer],
                                    [tl.Dense(d_model), mha_input_layer]),
                          mha_attention_layer,
                          mha_output_layer,
                          tl.Dense(d_model))

    return MHA_layer
```

#### Single Decoder Layer
```python
def DecoderLayer(d_model, d_ff, n_heads, dropout_rate, mode, ff_activation):

    """
    Create a single decoder layer

    Inputs
            d_model: <int> the dimenstion of embedding
            d_ff: <int> number of units in the feed-forward dense layer
            n_heads: <int> number of heads of attention
            dropout_rate: <float> the rate of dropout
            mode: <str> 'train' or 'eval' or 'predict'
            ff_activation: <trax layer> the feed-forward activation layer
    """

    ## Multi-Head Attention
    Multihead_Attention = MHA(d_model, n_heads, mode)

    ## Feed-Forward Block
    Feed_Forward_Block = [tl.LayerNorm(),
                          tl.Dense(d_ff),
                          ff_activation(),
                          tl.Dropout(rate = dropout_rate, mode = mode),
                          tl.Dense(d_model),
                          tl.Dropout(rate = dropout_rate, mode = mode)]

    ## Decoder Layer
    decoder_layer = [tl.Residual(tl.LayerNorm(),
                                 Multihead_Attention,
                                 tl.Dropout(rate = dropout_rate, mode = mode)),
                     tl.Residual(Feed_Forward_Block)]

    return decoder_layer
```

#### Transformer Language Model
```python
def TransformerLM(vocab_size = 33000, d_model = 512, d_ff = 2048, n_layers = 6, n_heads = 8,
                  dropout_rate = 0.1, max_len = 2*MAX_DATA_LENGTH, mode = 'train', ff_activation = tl.Relu):

    """
    Create the Transformer model

    Inputs
            vocab_size: <int> size of the vocabulary
            d_model: <int> dimensiton of embedding
            d_ff: <int> number of units of the feed-forward layer
            n_layer: <int> number of encoder/decoder layers
            n_heads: <int> number of attention heads
            dropout_rate: <float> rate of dropout layer
            max_len: <int> max length for positional encoding
            mode: <str> 'train', 'eval', or 'predict'
            ff_activation: <trax layer> the activation layer of the feed-forward layer
    """

    ## Positional Encoding
    Positional_Encoder = [tl.Embedding(vocab_size = vocab_size, d_feature = d_model),
                          tl.Dropout(rate = dropout_rate, mode = mode),
                          tl.PositionalEncoding(max_len = max_len, mode = mode)]

    ## Decoder
    Transformer_Decoder = [DecoderLayer(d_model, d_ff, n_heads, dropout_rate, mode, ff_activation) for _ in range(n_layers)]

    ## Transformer model
    model = tl.Serial(tl.ShiftRight(mode = mode),
                      Positional_Encoder,
                      Transformer_Decoder,
                      tl.LayerNorm(),
                      tl.Dense(vocab_size),
                      tl.LogSoftmax())

    return model
```



## References

This notebook was learned and modified from the assignment of the course Natural Language Processing with Attention Models on Coursera with the following amendments:

Instead of using a pre-trained model, I built and trained the summarizer from scratch

You can choose between 2 datasets (multi_news and cnn_dailymail)

I cleaned up and rewrote the part of building the Transformer model to make parameter tuning easier. The codes for the multi-head attention layer is more compact.

The sumarizer can be tested on the evlaution dataset or a custom article.

---
*Last updated: 11/05/2021*
