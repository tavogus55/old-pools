# Graph Pooling Experiments: Our Working Plan

**Notes from:** October 30, 2026  
**Aiming for:** January 2027

This is a shared reference for the researcher and developer to check that we're thinking about the same project. It isn't a formal submission. It brings together what we've discussed, the requirements the researcher supplied, and the suggestions we still need to talk through.

Open questions appear in bold next to the part they affect.

## 1. What we're trying to do

We want to compare the researcher's graph-pooling methods with existing methods on **both graph classification and graph regression**. We'll look at predictive quality, training efficiency, and inference efficiency using the same overall model and data splits.

The end result should be easy-to-compare tables for a paper and for discussion, with individual-run records behind them so we can check the numbers and recalculate summaries.

### What we expect, but still need to test

- The proposed methods should be around the comparison group's average or better for predictive performance, rather than consistently last.
- CountSketch with parameter 1 should be faster than uniform and Gaussian projection.
- The CountSketch variants and the other proposed methods should generally be faster than DiffPool and MinCut.

** (This needs clarification — Are these expectations or scientific truths?)** 

## 2. Who does what, and which information we trust

- **Researcher:** provides the theory, original methods, and scientific requirements; helps settle modeling/evaluation questions and scope trade-offs.
- **Developer:** implements the agreed setup, checks it, runs experiments, tracks resources, saves results, and prepares the tables.
- **Together:** decide what's feasible, interpret results, and settle the datasets, tuning budget, and schedule.

### Requirements versus suggestions

Some settings below come directly from the researcher's written instructions. Others were accepted by the developer during planning, but haven't necessarily been checked with the researcher. Suggestions are called out as suggestions; they shouldn't quietly become scientific decisions.

The researcher's explicit decisions take priority. Papers, repositories, and library documentation help us implement those decisions, but don't automatically define our experiment.

Only the **original local pooling modules** count as the intended method implementations. The current classifier and surrounding project were made for testing; they aren't the specification for the new experiment setup. No further prototype inspection was requested during planning. **(This needs clarification — C09: Should the latest implementation files provided remain fixed with no changes? Or will the researcher ask to edit if it encounters problems?)**

## 3. Which pooling methods we're comparing (for now)

| Existing methods | Proposed methods |
| --- | --- |
| DiffPool | Node-drop pooling (`ndrp`) |
| MinCut | Gaussian projection (`gaus`) |
| TopK | Uniform projection (`unif`) |
| SAGPool | Sparse CountSketch, parameter 1 |
| NDP | Sparse CountSketch, parameter 2 |
| Graclus | Sparse CountSketch, parameter 4 |

That's **twelve configurations**. NDP and `ndrp` are different methods.

Use sparse CountSketch as the main implementation because it's the faster version. Keep dense CountSketch available as a fallback/reference, not an additional configuration by default. If we substitute one for the other, record it rather than mixing results under one label.

### What the original proposed methods do

The source review reported the following behavior. These notes aren't correctness proofs or permission to change the algorithms:

- **Node dropping:** select nodes uniformly without replacement, keep their features and induced adjacency, and don't add projection scaling.
- **Gaussian:** use a dense random projection with independent entries of mean zero and variance `1/p`, where `p` is the output size.
- **Uniform:** use dense entries on `[-sqrt(3/p), sqrt(3/p)]`. This is projection, not uniform hard clustering.
- Gaussian and uniform calculate `X' = S X` and `A' = S A S^T`.
- **CountSketch:** parameter `q = 1/2/4` means the requested number of nonzero assignment coefficients per input-node column. Each input node contributes to up to `q` different output buckets; this isn't depth or pooling ratio.
- Sparse CountSketch scales signed coefficients by `1/sqrt(q_effective)`, performs the projected feature/adjacency products sparsely, sums duplicate edge contributions, and removes exact-zero cancellations. It can produce negative edge weights and self-loops.
- Reviewed CountSketch uses `p = max(1, floor(ratio * nodes))` and `q_effective = min(q, p)`. Node-drop/Gaussian/uniform use nearest rounding instead. Keep these differences and the tiny-graph behavior unless we explicitly agree to change them.

Preserve the original scaling, rounding, signed weights, empty buckets, and self-loops. The reviewed operators sample on every call, including evaluation. Training resampling with reproducible per-graph evaluation assignments was suggested, but not decided. **(This needs clarification — C08: Should validation/test resample, use a fixed assignment per graph/run/stage, or average several assignments? Should training keep resampling every forward pass?)** 

### A few baseline details we shouldn't overlook

The benchmark review found DiffPool, MinCut, TopK, and SAG, but didn't find NDP or Graclus in the inspected inventory. Use the official decimation-pooling reference for NDP and the PyG reference for Graclus.

NDP/Graclus don't necessarily halve the node count. We've agreed to preserve their native sizes when ratio 0.5 isn't supported. **(This needs clarification — C07: What exact NDP/Graclus adapter settings, edge weights, feature aggregation, and coarse-edge construction should we use?)** Don't change their algorithms just to force exact halving.

We won't use pooling auxiliary losses. That makes DiffPool/MinCut **task-loss-only variants**, not unchanged reproductions of their original objectives. **(This needs clarification — C10: Can the researcher confirm this choice and the wording we'll use to explain it in the results?)** The developer has accepted the distinction.

## 4. Which datasets we'll start with

We want a mix of small and large graphs. The target is at least **twelve classification datasets and six or seven regression datasets**. We can choose additions/replacements based on runtime and the deadline.

### Classification

Start with:

- `ogbg-molpcba`
- PROTEINS
- NCI1
- NCI109
- MUTAG
- D&D
- IMDB-BINARY, also called IMDB-B
- IMDB-MULTI, also called IMDB-M
- COLLAB

`ogbg-molhiv` is another candidate. If it's too slow or large, we can replace it for feasibility. We have nine initial datasets, or ten with molhiv, so even with molhiv we need at least two more to reach twelve. **(This needs clarification — C02: Which datasets are mandatory, which ones should we try next, and what order should we use for substitutions if compute gets tight?)**

### Regression

Start with QM7, QM8, BACE, ESOL, FreeSolv, and Lipophilicity. That's six. **(This needs clarification — C02: Do we need a seventh regression dataset for the final results, or is six enough?)**

BACE means its **regression** evaluation, not classification. We will use its regression target.

### What goes into each graph

We'll record each dataset's exact identifier, version, task, and targets. Each method should receive the same agreed input representation. **(This needs clarification — C12: Which node features, categorical encodings, edge weights/attributes, and directedness handling should we use? What if a dataset has no node features, missing labels, or grouping constraints?)** Don't inherit unit-weight assumptions or silently discard information because the prototype did.

## 5. What the shared model looks like

Use the same GraphConv backbone for every method. The original specification used GCNConv; GraphConv was chosen during planning because of the signed-weight concern raised for CountSketch. We shouldn't switch backbones per method after seeing results.

The intended sequence is:

1. Input node features.
2. GraphConv from input width to 32, then ReLU and dropout.
3. First pooling stage, ratio 0.5 where supported.
4. GraphConv from 32 to 32, then ReLU and dropout.
5. Second pooling stage, ratio 0.5 where supported.
6. GraphConv from 32 to 32, then ReLU and dropout.
7. Concatenate the element-wise global mean and global max: 32 + 32 = 64 values.
8. A task-specific linear output layer.

The layer family is settled, but not all of its behavior. Sum aggregation with signed weights preserved was a suggestion, not a decision. **(This needs clarification — C04: Which GraphConv aggregation, edge-weight handling, normalization, and self-loop policy should we use?)**

DiffPool/MinCut need assignment networks as well as pooling operators. **(This needs clarification — C06: What should those networks look like? How do we set and round cluster counts, handle padding masks, and account for extra parameters? Does output size depend on each graph's valid node count or another size?)** Keep SAGPool's internal scoring network separate from the common backbone when specifying its configuration.

### Output layer and training loss

| Task | Current plan |
| --- | --- |
| Single-label classification | Linear(64, number of classes). The original specification used LogSoftmax. Logits with cross-entropy was accepted as a proposal; LogSoftmax with NLL is an equivalent possible formulation. Use one consistent choice. |
| Binary/multilabel classification | Appropriate output logits and binary-cross-entropy, accepted as a proposal during planning. |
| Regression | Linear(64, 1) or Linear(64, number of targets), with no activation, as supplied by the researcher. Train with MSE on normalized targets, as accepted in planning. |

There are no pooling auxiliary losses. **(This needs clarification — C13: Can we confirm the proposed classification heads/losses, missing-label rules, and evaluator behavior for each dataset?)**

### Where dropout goes

Dropout has a default probability of 0.2 and is applied immediately after ReLU in each of the three GraphConv layers. In the first two layers, it comes before pooling; in the third, before readout. No additional dropout is specified for assignment/scoring branches or the final classifier. This replaces the earlier readout-only suggestion.

## 6. How we'll split the data and train

For every dataset:

1. Make ten outer folds, shared by all methods.
2. Hold one fold out for testing and use the other nine as the development pool.
3. Take 10% of that pool for validation. The rest is the actual optimization training subset.
4. Keep those fold/validation assignments fixed across methods and final training seeds.
5. Train independently with ten seeds, using the same seed identities across methods.
6. Evaluate each trained model from its selected checkpoint.

That gives **ten folds x ten seeds = 100 trained-model results per method/dataset**. This is custom cross-validation, not the official dataset benchmark split, so we'll say that in the results.

**(This needs clarification — C11: How should we create the folds and validation subsets: class-stratified, random, scaffold-based, or grouped? Which split-generation seeds should we use, and how should multilabel datasets be handled?)** Fixed shared assignments are settled; their construction isn't.

### Default training settings

| Setting | Value |
| --- | --- |
| Optimizer | Adam |
| Learning rate | 0.001 (`1e-3`, not the earlier typo `1e03`) |
| Weight decay | 0.0001 (`1e-4`) |
| Batch size | 128 |
| Dropout | 0.2 |
| Node embedding width | 32 |
| Pooling ratio | 0.5 when supported; native original behavior otherwise |
| Maximum epochs | 2,000 |
| Early-stopping patience | 50 consecutive epochs without a strict validation improvement |

Use the chosen layers’ standard initialization and Adam’s default settings except for the explicitly specified learning rate and weight decay. Keep these conventions consistent across methods and record the resolved setting
### Early stopping and checkpoint choice

The accepted proposal is to maximize validation accuracy for ordinary single-label classification, maximize AP/ROC-AUC where designated, and minimize validation MSE for regression. Only a strict improvement resets patience; ties don't. Test the best-validation checkpoint, not automatically the last epoch. Never use the outer test scores to choose checkpoints or hyperparameters. These choices should be included when we review the plan together.

### Small experiments versus final results

Pilots can use two or three training seeds, but still keep ten folds. Final evaluation targets ten seeds. Label pilots clearly rather than treating them as final evidence. Any change to the final repetition count needs a separate discussion; see the fallback question in Section 10.

## 7. How regression targets are normalized

The researcher supplied these rules:

1. Normalize every target dimension separately using training mean and standard deviation.
2. Train against those normalized targets.
3. Convert predictions back to their original units before calculating validation/test metrics.
4. Never calculate normalization statistics from validation or test data.

The suggested interpretation is to fit statistics only on the actual optimization subset, after removing validation, separately for each outer fold, then reuse those statistics unchanged. **(This needs clarification — C14: Is that the exact intended training boundary? What should happen for zero/near-zero target variance or missing targets? Which standard-deviation convention should we use?)** The training-only rule is clear; these details still need checking.

**(This needs clarification — C14: How should we combine original-scale errors across multiple targets, especially when their scales differ, including the MSE used for validation selection?)** Averaging per-target RMSE isn't necessarily the same calculation as taking the square root of an aggregated MSE.

## 8. Which predictive metrics we'll report

| Task/dataset | Report | Use for validation selection |
| --- | --- | --- |
| Ordinary single-label classification | Accuracy and Macro-F1. Micro-F1 may be equivalent to accuracy in the applicable single-label setting. | Accuracy, under the accepted proposal. |
| `ogbg-molpcba` | Average precision, with the right task/label handling. | AP. |
| `ogbg-molhiv`, if included | ROC-AUC. | ROC-AUC. |
| Regression | MSE, MAE, and RMSE in original target units. | MSE. |

**(This needs clarification — C13: Which exact evaluator definitions should we follow, and what happens with missing labels or undefined class metrics?)** Multi-target regression aggregation is the question in Section 7. These tables don't establish a formal equivalence or non-inferiority claim; that question is in Section 1.

## 9. How we'll measure speed and memory

### Training

- Training time per epoch.
- Total number of training epochs, accounting for early stopping.
- Total training time for a final model.
- Peak training GPU memory.

### Inference

- Total model-execution time over the test set.
- Peak inference GPU memory.
- Throughput: test graphs processed per second within the stated timing boundary.

### What the measurements include

Measure **model execution only**, consistently across methods. Preprocessing is excluded from the requested runtime results; separately reporting preprocessing cost was suggested and declined. Don't include loading or device transfers for one method while excluding them for another.

**(This needs clarification — C18: Does training time include validation? How should we summarize epoch times? What warm-up, device synchronization, inference repetitions, transfer boundaries, and device-concurrency rules should we use? For peak memory, what gets reset and should we measure allocated or reserved memory?)** These details aren't decided defaults.

Keep final-model training cost separate from hyperparameter-search cost. Retain search cost and trial counts so we can still track the full budget.

Tuning can choose different batch sizes for different methods. We should show the chosen batch sizes beside the results. **(This needs clarification — C19: Is speed at each method's tuned configuration enough, or do we also want a separate comparison at a shared batch size?)** The extra controlled experiment hasn't been agreed as required work.

For throughput summaries, mean per-run throughput and total graphs divided by total time aren't the same statistic. **(This needs clarification — C23: Which summary should appear in the tables, and how should it be labeled?)**

## 10. How we'll tune hyperparameters

We prefer tuned results in the final tables. Default-setting results are still useful and could be a fallback. **(This needs clarification — C20: If tuning doesn't fit the deadline or allocation, would default-only tables be an acceptable delivery? What scope or final-seed reductions would be acceptable instead?)** Don't assume pilots or defaults fulfill the final promise without discussing it.

### General parameters supplied by the researcher

Use Adam and tune coordinates in this order:

| Order | Parameter | Values to try |
| --- | --- | --- |
| 1 | Learning rate | 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005 |
| 2 | Weight decay | 0.01, 0.001, 0.0001, 0.00001 |
| 3 | Dropout | 0.0, 0.2, 0.5 |
| 4 | Batch size | 32, 64, 128, 256 |

Keep fixed node dimensions and pooling ratios fixed unless we agree otherwise. Record all selected parameters for the paper.

### What we mean by coordinate search so far

The proposed procedure is: start at the defaults, vary one parameter while keeping the others fixed, choose using validation performance, and carry that choice to the next parameter. This was accepted as a proposal, not a complete search specification.

**(This needs clarification — C15: Do we make one pass through the coordinates or repeat until a stopping condition? How do we break ties, order candidates, reuse unchanged trials, set trial epoch limits, and handle failed trials?)**

One search per method/dataset/outer fold, followed by ten freshly trained final-seed models, was suggested but not confirmed. **(This needs clarification — C15/C17: Is that where tuning should happen? Which seeds/repeats should select the settings, should they be separate from final evaluation seeds, and should methods receive equal trial counts, equal compute time, or another budget?)** Outer test performance must never select the settings.

### Method-specific parameters

The supplied reference is Table A.4 on printed page 27 of the graph-pooling benchmark. Review of arXiv v5 reported:

| Method | Table A.4 search |
| --- | --- |
| TopK/SAG | Ratios 0.1, 0.3, 0.5, 0.7, 0.9. Don't search these if ratio stays fixed at 0.5. |
| DiffPool | No method-specific coordinate listed. |
| MinCut | Temperature 1.0, 1.5, 1.8, 2.0. |

**(This needs clarification — C16: Should MinCut temperature be tuned, and where does it go in the coordinate order? Which parameters, if any, should be tuned for NDP, Graclus, or the proposed methods?)** The table doesn't supply complete settings for those methods. Don't add arbitrary library options or auxiliary-loss coefficients to the search. CountSketch 1/2/4 remain separate reported configurations unless we explicitly change that plan.

## 11. What we'll save and hand over

### Individual-run records

Save machine-readable records so we can regenerate summaries and trace a table entry back to its experiments. The suggested fields below aren't a fixed schema yet:

- Run ID and phase: pilot, default, tuning, or final.
- Dataset/version, method/configuration, and source revision.
- Fold, training seed, split ID, and validation assignment ID.
- Full settings, chosen checkpoint/epoch, and completed epochs.
- Predictive metrics, timing, memory, batch size, and device/environment ID.
- Completion status, failure reason, and retry/resume history if implemented.
- Regression normalization details and a reference to the selected tuning configuration.

Keep tuning trials and selected settings separate from final runs. Calculate summaries from unrounded run records, not already-rounded table numbers.

### How we'll aggregate

- Report mean and **sample standard deviation** across the 100 final fold/seed outcomes.
- Keep per-fold summaries and all original outcomes.
- Show completed/expected counts, including failed or partial experiments.
- Never silently drop failures or make a partial average look complete.
- Label pilots with their actual seed counts.
- Don't treat 100 outcomes as proof of statistical independence or formal significance.

**(This needs clarification — C23: What units, precision, rounding, and aggregation conventions should the tables use? How should incomplete comparisons and rankings be displayed?)**

### Files we'll share

1. **Excel:** aggregated tables for discussion and presentation.
2. **LaTeX:** paper-ready tables.
3. **CSV:** machine-readable individual runs and recorded selected hyperparameters.
4. Short explanation notes covering custom splits, task-loss-only baselines, native pooling-size exceptions, partial results, and timing boundaries.

Predictive quality, training, and inference may use separate table groups. **(This needs clarification — C23: What table layout and formatting would be most useful for the paper and our discussions?)** A presentation slide deck itself isn't a committed deliverable.

## 12. A sensible order for the work

This is a suggested sequence, not a decision to start implementation:

1. **Clear up the scientific questions.** Settle the backbone, pooling details, inputs, splits, and normalization first. Then tuning and measurement; then scope, recovery, schedule, fallback, and presentation.
2. **Implement and check the agreed behavior.** Check method adapters, splits, metrics, normalization, and timing rather than copying prototype assumptions.
3. **Run small pilots.** Use two/three seeds and ten folds on representative small/large cases to estimate resource needs.
4. **Run default-setting experiments.** Build a fixed-configuration reference without choosing settings from test scores.
5. **Run coordinate search.** Follow the agreed bounded procedure and save the selection history.
6. **Run final evaluation.** Train selected configurations with ten folds and ten seeds.
7. **Check and share results.** Verify completeness, generate tables, explain limitations, and review together.


## 13. Time and computing budget

The full matrix is large:

- `12 methods x (12 classification + 6 regression datasets) x 10 folds x 10 seeds` = **21,600 final training runs**.
- With seven regression datasets, that's **22,800**.
- Those totals exclude pilots, tuning, retries, and extra datasets. Each run allows up to 2,000 epochs, with early stopping.

We haven't measured feasibility or established a reliable completion-time estimate. Use pilots to estimate the workload and choose priorities together; don't quietly reduce the final seeds or promised scope.

We're aiming for January 2027. During the October 30 discussion, the exact date was expected to be discussed the following week, but no date was committed. **(This needs clarification — C01: What is the final-table date, which intermediate checkpoints matter, and how much time should we leave for reruns and review?)**

## 14. What happens when a run fails

Record out-of-memory errors, timeouts, invalid outputs, and other failures visibly. Don't quietly change settings until a method succeeds. Keep completion counts alongside results so a partial comparison doesn't look finished.

Tracking runs independently and skipping completed results were suggested. **(This needs clarification — C21: Should an interrupted run restart or resume? How many retries are allowed, which recovery actions can be automatic, and should one failed configuration stop a larger batch?)**

If we choose resume, it needs a defined full-state checkpoint: model, optimizer, early-stopping state, random states, and scheduler state if used. Don't change scientific settings silently during recovery.

### Things to watch out for

| Potential problem | How we'll avoid or discuss it |
| --- | --- |
| Too much work for the allocation/deadline | Pilot estimates, agreed priorities/fallback, no silent final-seed reduction. |
| Signed pooled weights don't match convolution assumptions | Settle the exact common GraphConv behavior before large runs. |
| Modified baselines look like original reproductions | Preserve native algorithms and explain task-loss-only objectives. |
| Dataset names hide different targets or inputs | Record exact identifiers and features, especially BACE regression. |
| Information leaks from validation/test | Shared recorded splits, training-only normalization, validation-only selection. |
| Random pooling changes what evaluation measures | Agree on sampling and seed policy. |
| Different timing/batch settings hide an unfair comparison | Show selected settings and define timing/memory boundaries. |
| Incomplete results look complete | Keep counts, failure reasons, pilot labels, and raw records. |
| Prototype behavior becomes an accidental requirement | Use original pooling modules, but specify experiment infrastructure separately. |
| Tuning becomes open-ended | Set placement, seeds, stopping rules, and budget before starting. |


## 15. References 

1. Benchmark repository: https://github.com/goose315/Graph_Pooling_Benchmark/tree/main
2. Reviewed benchmark paper: https://arxiv.org/abs/2406.09031v5
3. Table A.4, printed page 27: https://arxiv.org/pdf/2406.09031v5
4. Official NDP: https://github.com/danielegrattarola/decimation-pooling
5. PyG Graclus: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.pool.graclus.html
6. PyG DiffPool: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.dense.dense_diff_pool.html
7. PyG MinCut: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.dense.dense_mincut_pool.html
8. PyG TopK: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.pool.TopKPooling.html
9. PyG SAGPool: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.pool.SAGPooling.html
10. Researcher-provided dropout example: https://github.com/tkipf/pygcn/blob/master/pygcn/models.py
11. Researcher-provided optimizer example: https://github.com/tkipf/pygcn/blob/master/pygcn/train.py
12. OGB task/metric reference: https://ogb.stanford.edu/docs/graphprop/

