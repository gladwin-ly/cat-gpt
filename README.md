<div align="center">

<h1>🐱 CatGPT 🐱</h1>

<p><b>A GPT-2 class transformer, implemented entirely in Scratch.</b></p>

<p><i>By <a href="https://github.com/gladwin-ly">Gladwin Ly</a>.</i></p>

<p>
  <img src="https://img.shields.io/badge/built%20in-Scratch-FFAB19?style=for-the-badge&logo=scratch&logoColor=white" alt="Built in Scratch">
  <img src="https://img.shields.io/badge/blocks-2647-4C97FF?style=for-the-badge" alt="2647 blocks">
  <img src="https://img.shields.io/badge/architecture-GPT--2%20%2F%20GPT--Neo-9966FF?style=for-the-badge" alt="GPT-2 / GPT-Neo">
</p>

<p>
  <a href="TODO_SCRATCH_LINK"><b>▶ Try it</b></a>
  &nbsp;·&nbsp;
  <a href="TODO_YOUTUBE_LINK"><b>📺 Watch the explainer</b></a>
  &nbsp;·&nbsp;
  <a href="#-code"><b>🔧 How it works</b></a>
</p>

<img src="TODO_demo.gif" alt="CatGPT generating text" width="620">

</div>

<br>

> The concept of a LLM (large language model) has only just recently been deeply explored within the scope of technological innovations. Now, it's in Scratch.

<br>

---

<details open>
<summary><h2>📖 Introduction</h2></summary>

<br>

Vaswani et al.'s paper **[Attention is All You Need](https://arxiv.org/pdf/1706.03762)** proposed the transformer model, a complex mechanism primarily meant for translation and transduction models—the authors of which were probably blissfully unaware of the innovation they sparked in the world of artificial intelligence. And now, nearly a decade later, that mechanism is **fully ported to Scratch**.

CatGPT's end-to-end mechanism, from tokenization to our final logit search, uses a mere `2647` blocks, something that was worked on to be extremely optimized.

<br>

<table>
<tr>
<td valign="top" width="50%">

#### 🧮 CatGPT contains these functions:

`matrix addition`
`matrix multiplication`
`hadamard multiplication`
`geLU`
`softmax`
`matrix dequantization`
`matrix transposition`

</td>
<td valign="top" width="50%">

#### ⚡ ...with the ability to perform:

`tokenization`
`layer normalization`
`multi-head attention (not concurrent)`
`multi-layer perceptron (feed forward network)`
`compatibility with gpt-neo and gpt-2 architecture`

</td>
</tr>
</table>

These functions allow the wide range of processing that the project has to do to produce a valid output.

</details>

---

<details open>
<summary><h2>💻 Code</h2></summary>

<br>

### Tokenization

<kbd>118 blocks</kbd> &nbsp; <kbd>Pain level: 5</kbd>

Tokenization gets its own special sprite because it was developed first and I may have been too lazy to move it to the matrix operations sprite. But, consequently, there are no matrix operations involved in tokenization—which is simply the act of separating the user's inputs into tokens.

This was done by building a "token" variable by one letter of the user's input at a time, matching it with the imported tokens database every time until it was the longest possible token that could be achieved before moving on to the next part of the input.

In theory this should have been very little pain, but there were definitely heaps of pre-processing issues and issues with lowercase/uppercase tokens (Scratch comparisons are case-_insensitive_, meaning I had to do a funny little costume tactic (costume comparisons are, in contrast, case sensitive operations) in order to fix this issue).

<br>

### LayerNorm

<kbd>471 blocks — 3 functions</kbd> &nbsp; <kbd>Pain level: 3</kbd>

LayerNorm is simply just a formula:

<table align="center"><tr><td>

$$y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} \cdot \gamma + \beta$$

</td></tr></table>

If you consider making the matrix operations efficient (accessing matrix rows, hadamard products, addition, matmul) part of the process, I would raise the pain level, as this is where the first matrix operations really are—but after that, layerNorm (and consequently the post-attention and post-FFN layerNorms) were not too difficult to pull off.

<br>

### Multi-headed attention

<kbd>483 blocks</kbd> &nbsp; <kbd>Pain level: 11</kbd>

<table align="center"><tr><td>

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V$$

$$\mathrm{MultiHead}(Q, K, V) = \mathrm{Concat}(\mathrm{head}_1, \ldots, \mathrm{head}_h)\,W^{O}$$

$$\mathrm{head}_i = \mathrm{Attention}(QW_i^{Q},\, KW_i^{K},\, VW_i^{V})$$

</td></tr></table>

The lack of proper Scratch debugging really starts to get me here. Of course I knew this would be the most abysmal part of the project, but there were so many bugs, including a leading space that appeared sometimes in front of certain matrices, derailing all my operations.

> NOTE: I literally put a band-aid on this bug by just removing a leading space from each row if it exists: it's one of life's sad truths, but I still don't know (and don't intend to find out) the true reason behind this bug.

Lots of work was done to optimize this mechanism, and I'm sure there can be much more work that I missed (most likely with issues found in finding a value at `(i,j)` of a matrix) but it works well enough for my models that I'm fine leaving it how it is.

<br>

### MLP/FFN

<kbd>126 blocks</kbd> &nbsp; <kbd>Pain level: 4</kbd>

<table align="center"><tr><td>

$$\mathrm{FFN}(x) = \mathrm{GeLU}(xW_1 + b_1)\,W_2 + b_2$$

$$\mathrm{GeLU}(x) = x \cdot \Phi(x) \approx 0.5x\left(1 + \tanh\!\left[\sqrt{\tfrac{2}{\pi}}\left(x + 0.044715x^{3}\right)\right]\right)$$

</td></tr></table>

I included GeLU as part of the block count. This wasn't very bad in Scratch, despite being a major part of the transformer architecture. Like LayerNorm, it follows a simple formula that becomes easy to implement once all the major operations were completed and optimized.

<br>

---

<div align="center">

That's basically it: most of the difficulties lied in optimization, dequantization and preprocessing into Scratch. Actual implementations were shaky at best.

</div>

</details>

<details open>
<summary><h2>🔎 Specs</h2></summary>

<br>

Weights were ported from [TinyStories](TODO_MODEL_LINK) for GPT-Neo architecture and from [DialoGPT](https://huggingface.co/microsoft/DialoGPT-small) for GPT-2 architecture (largely untested). Models were pretrained in Python.

<br>

<div align="center">

| | TinyStories (GPT-Neo) | DialoGPT (GPT-2) |
|:---|:---:|:---:|
| **Parameters** | `1M-33M` | `117M` |
| **Layers** | `1-8` | `12` |
| **Attention heads** | `16` | `12` |
| **Embedding dim** | `64` | `768` |
| **Vocab size** | `16k-50k` | `16k` |
| **Context length** | `256` | `1024` |
| **Positional encoding** | learned | learned |
| **Activation** | GeLU | GeLU |

</div>

</details>
