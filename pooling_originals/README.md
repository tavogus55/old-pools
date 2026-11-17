# Research-source provenance

These operators are used inside the project's existing three-convolution,
two-pooling backbone, not as replacements for the entire upstream networks.
Graph-level logits and regression outputs use the existing output head and
training/evaluation code.

## Source snapshots

Benchmark repository: https://github.com/goose315/Graph_Pooling_Benchmark

Pinned benchmark commit: `f9e4670f8a4c785acd6268ff64d1797fccdde233`.

| Local file | Upstream implementation |
|---|---|
| `asym.py` | `AsymCheegerCutPooling.ipynb`, first `AsymCheegerCutPool` definition |
| `sep.py` | `SEP/SEPG/sep.py`, `SEPooling` |
| `sep_tree.py` | `SEP/SEPG/utils.py`, coding-tree functions and `PartitionTree` |
| `sep_transform.py` | `SEP/SEPG/trans_graph.py`, tree conversion functions |

PARS is absent from the benchmark snapshot. Its original author implementation
is https://github.com/LUMIA-Group/GraphParsingNetworks, pinned at
`cd0c5d8b892cb05f26e0d95c4f39669b63e0c0ea`. `pars.py` contains the `MLP` and
`ParsingNet_GPU` definitions from `layer.py`. The accompanying author licenses
are retained in `PARS_LICENSE` and `SEP_LICENSE`.

## Minimal source changes

- Extract reusable operators; omit notebook training cells, unrelated network
  definitions, dataset loading, hard-coded working directories, multiprocessing
  entry points, and experiment-tracking imports.
- Adjust imports to make the extracted modules self-contained and package-local.
- PARS: implement the author's one-dimensional community mean with
  `torch_scatter.scatter(..., dim=0, reduce="mean")`. This is equivalent to
  sorting plus the original CSR mean, without redundant sorting. PyG 2.1's
  graph readout is not a drop-in replacement for one-dimensional scores.
- PARS: remove `torch.set_num_threads(1)` from every forward. Pooling must not
  silently and permanently change the entire training process's thread count.
- SEP: pass the supplied bipartite `size` into `propagate` rather than ignoring
  it. Its parent dimension is generally smaller than its child dimension.

All other adaptations live in `../pooling_adapters.py`.

## Integration semantics

### AsymCheegerCut (`asym`)

Preserve the benchmark's `[hidden, 128, 128, clusters]` assignment MLP,
softmax, `S.T @ X`, `S.T @ A @ S`, total variation, and asymmetric balance
formulas. Cluster counts follow this pipeline's two-stage `--pratio` schedule.
Input masks are passed to the original operator. Unlike DiffPool, the pooled
adjacency is not replaced with a normalized variant.

The supplied notebook and this pipeline optimize the supervised task loss
only, without adding pooling regularizers. The pipeline therefore skips
Asym's **unused auxiliary diagnostics**, returning zero auxiliary placeholders.
This avoids computing an expensive pairwise loss that cannot affect either
the predictions or task-loss gradients. The standalone adapter defaults to
computing the original losses. Enable diagnostics on a pipeline model with:

```python
model.pool1.asym.compute_auxiliary_loss = True
model.pool2.asym.compute_auxiliary_loss = True
```

Enabled diagnostics replace the original `B*N*N*K` L1 broadcast with exact
batched `torch.cdist(..., p=1)`, chunked to limit temporary memory. The
normalizer is guarded for edgeless graphs. For one output cluster the original
balance expression divides by zero, so its degenerate loss is defined as zero.
Otherwise the original formulas, including their mask/padding conventions,
are unchanged. Enabling diagnostics does not add them to the training loss.

### Graph Parsing Networks (`pars`)

Use the original learnable edge MLP, undirected parsing-edge dropout,
dominant-edge community parsing, isolated-node treatment, binary quotient
adjacency, and differentiable `node_score * link_counts` gating of `S.T @ X`.
The original CPU community-discovery loop remains intact; the learned network,
feature contraction, and score gradients remain on the model device. Input
and quotient topology must be undirected, as assumed by the author code.

The former local PARS approximation is removed. In particular, the original
uses undirected dropout and both community-score and link-count factors;
results from the old approximation are not the same model configuration.

### Structural Entropy Pooling, SEP-G (`sep`)

Use the original `PartitionTree`/tree conversion at depth three, matching two
pooling stages before graph readout. SEP-G is the graph-task variant; SEP-N's
node-task architecture is not substituted. Disconnected graphs use the
original component-aware conversion. A one-node graph uses identity
assignments at both levels, avoiding the source tree builder's undefined
single-node root.

Precompute and persist only topology-derived assignments, before splitting
and training, under `data/sep_trees/<dataset>_depth3_v1.pt`. Cache keys hash
the node count and sorted undirected edges, never features or targets. Graphs
with identical topology reuse assignments. Concurrent processes write through
distinct temporary files and atomically replace the cache. Changing topology
invalidates that graph's entry. Delete the cache when changing tree algorithms.

`SEPData.__inc__` offsets each hierarchy's community IDs during ordinary PyG
batching, including shuffled/subset loaders. The original child-to-parent sum
uses its supplied bipartite dimensions. Sparse contraction and deduplication
produce the same binary quotient edges as upstream NetworkX, without running
NetworkX or building coding trees in a forward pass.

PARS and SEP determine community counts from topology; `--pratio` is accepted
but does not override their algorithms. Their original quotient graphs are
unweighted. Scalar input edge weights are still used by the first shared
convolution, but are not propagated as weighted quotient edges. Asym retains
weighted dense contraction. All three use raw output heads for classification
and single-/multi-target regression rather than forcing log-softmax outputs.
