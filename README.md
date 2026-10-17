# Activation Steering and Latent Adversarial Robustness for LLM Safety

This repository studies the internal mechanisms underlying refusal failures in language models.

The project uses activation steering as a causal tool to test how refusal-related representations affect model behavior, whether jailbreak-induced safety failures can be explained by movement away from locally supported activation geometry, and whether failure-inducing latent states can be reused as adversarial training signals.

The work began with a mechanistic question:

> What properties of internal activation changes determine whether a model maintains or loses refusal behavior under adversarial perturbations?

The project has progressed through three stages:

1. **Mechanistic probing** of refusal-related activation structure
2. **Causal intervention and geometry analysis** using activation steering and random latent perturbations
3. **Latent adversarial fine-tuning**, where internal states that induce refusal failure are used as training-time adversaries

---

## Research Progression

```text
Characterize jailbreak activations
            ↓
Identify refusal-associated directions
            ↓
Causally intervene with activation steering
            ↓
Test whether failures are explained by off-manifold movement
            ↓
Stress-test with random latent perturbations
            ↓
Analyze final refusal-axis position
            ↓
Replicate across model families
            ↓
Train against failure-inducing latent states
```

The overall goal is to move from **understanding internal safety representations**, to **causally manipulating them**, and finally toward **using those failures to improve adversarial robustness**.

---

## 1. Refusal Representations and Activation Steering

The first question was whether refusal-related structure in activation space is simply correlated with model behavior or is causally involved in producing refusal.

For a hidden activation `h` and a learned refusal-associated direction `r`, I intervene using:

```text
h' = h + alpha * r
```

By varying the sign and magnitude of `alpha`, the model can be pushed toward or away from refusal-associated regions of activation space.

This makes activation steering a causal probe rather than only an analysis tool.

### OLMo

Using `OLMo-2-0425-1B-SFT`, I estimated a refusal direction from harmful prompts that the model either refused or answered.

Moving activations away from the refusal direction reduced refusal behavior, while steering in the opposite direction preserved refusal.

This provided initial evidence that the learned direction captures behaviorally meaningful internal structure.

### Qwen

I then replicated the intervention on `Qwen2.5-1.5B-Instruct`.

Because successful jailbreaks produced a highly imbalanced refusal split, I constructed a matched refusal-associated transition direction using paired base prompts and their corresponding successful jailbreak variants.

Steering successful jailbreak activations toward the refusal-associated side restored refusal from:

```text
0% -> 95%
```

at the strongest tested intervention.

This suggests that refusal-associated activation structure is causally meaningful across more than one model family.

Relevant code:

```text
build_refusal_direction_layer8.py
validate_refusal_direction.py
steering_validation_20.py

cross_model_pipeline/pipeline/02_matched_refusal_direction.py
cross_model_pipeline/pipeline/03_qwen_targeted_steering.py
```

---

## 2. Are Safety Failures Explained by Off-Manifold Movement?

The next hypothesis was that jailbreaks or steering-induced failures might occur because model activations leave the locally supported representation geometry.

This motivated a distinction between:

- **out-of-distribution inputs**
- **off-manifold activation states**

These are not necessarily the same.

A prompt can be unusual at the input level while still producing an activation supported by nearby model representations. Conversely, an internal intervention can move an activation far away from its local neighborhood without changing the input distribution.

To study this, I built a reference activation bank and estimated a local tangent space around each test activation using:

1. cosine nearest neighbors
2. local PCA
3. enough principal components to explain 90% of neighborhood variance

For an intervention displacement:

```text
delta_h = h_post - h_pre
```

I decompose it as:

```text
delta_h = delta_h_tangent + delta_h_normal
```

where:

- `delta_h_tangent` is movement within the local tangent space
- `delta_h_normal` is movement normal to that space

I then measure:

- tangent norm
- normal norm
- normal fraction
- pre-intervention residual distance
- post-intervention residual distance
- residual increase

Relevant code:

```text
build_layer8_reference_bank.py
decompose_refusal_direction_local_pca.py
compare_geometry_by_behavior.py
```

---

## 3. A Useful Negative Result

One initial hypothesis was:

> If steering causes refusal failures by pushing activations off-manifold, then non-refusal runs should show larger normal displacement than refusal runs.

That is not what I observed.

For Qwen, at the primary random-steering setting:

```text
Median normal norm

Refusal:               41.93
Coherent non-refusal:  41.85
```

The corresponding residual increases were also nearly identical.

This suggests that **simple local-PCA off-manifold magnitude is not sufficient to explain which perturbations cause refusal failure**.

This negative result changed the direction of the project.

Rather than treating distance from the local activation manifold as the main explanatory variable, I began testing whether movement along specific behaviorally meaningful directions was more informative.

---

## 4. Refusal-Axis Position

The next analysis focused on where the model's actual hidden state ends up after steering.

For each run:

```text
s_pre  = h_pre dot r
s_post = h_post dot r
delta  = s_post - s_pre
```

This separates two different questions:

1. Is the sampled steering vector itself aligned with the refusal direction?
2. Where does the model's actual activation end up after the intervention?

For Qwen, random-vector alignment alone did not strongly predict failure.

However, the final post-steering activation position showed a modest association with refusal behavior: runs that still refused ended somewhat more aligned with the refusal-associated axis.

This suggests that **the final internal state may be more informative than the nominal direction of the intervention alone**.

Relevant code:

```text
cross_model_pipeline/pipeline/07_qwen_refusal_axis_positions.py
```

---

## 5. Random Latent Perturbations as a Safety Stress Test

To test whether refusal failure requires precise steering along a known refusal direction, I also applied random activation-space perturbations inspired by Rogue Scalpel.

For a random unit vector `v`:

```text
h' = h + alpha * v
```

where `alpha` is scaled relative to the natural activation norm at the intervention layer.

On Qwen, the selected prompts began at:

```text
100% refusal
```

At the primary random-perturbation setting, refusal dropped to:

```text
9.5%
```

while stronger perturbation settings produced more degeneration.

This creates an important contrast:

- targeted activation steering shows that refusal-related directions are causally meaningful
- random latent perturbations can still cause widespread refusal failures
- simple off-manifold magnitude does not distinguish those failures
- final activation position along refusal-associated structure appears more informative

Relevant code:

```text
rogue_scalpel_layer8_focused.py

cross_model_pipeline/pipeline/04_qwen_rogue_scalpel.py
cross_model_pipeline/pipeline/06_qwen_alignment_geometry.py
```

---

## 6. Cross-Model Replication

The experiments were first developed on OLMo and then reorganized into a reusable cross-model pipeline.

Current models:

- `OLMo-2-0425-1B-SFT`
- `Qwen2.5-1.5B-Instruct`

Additional model-family replication is in progress.

The reusable pipeline includes:

```text
01_baseline.py
02_matched_refusal_direction.py
03_qwen_targeted_steering.py
04_qwen_rogue_scalpel.py
05_qwen_reference_bank.py
06_qwen_alignment_geometry.py
07_qwen_refusal_axis_positions.py
```

with shared configuration and model-loading utilities.

The goal is to distinguish mechanisms that are model-specific from effects that persist across model families.

---

## 7. Current Direction: Latent Adversarial Fine-Tuning

The project is now moving from **diagnosis to mitigation**.

The random-steering experiments provide latent perturbations that empirically induce refusal failures.

Instead of only analyzing those states, the next step is to reuse them during fine-tuning:

```text
harmful prompt
      ↓
apply failure-inducing activation perturbation
      ↓
model enters a non-refusal / unsafe latent state
      ↓
train the model to produce the safe response anyway
```

The central question is:

> Can training against latent states that are known to cause safety failures improve robustness to unseen jailbreaks and unseen internal perturbations?

The first planned comparison is between:

- standard refusal fine-tuning
- training on failure-inducing Rogue-Scalpel perturbations
- targeted refusal-direction perturbations
- PGD-based targeted latent adversarial training

Later experiments may also combine multiple latent adversary generators.

Evaluation will focus on transfer to:

- held-out random steering directions
- unseen jailbreak prompts
- stronger or shifted latent interventions
- behavioral refusal robustness
- benign capability and over-refusal

The key idea is to use internal failures not only as probes of model behavior, but as **training signals for adversarial robustness**.

---

## Current Takeaway

The experiments so far support a more nuanced picture than a simple "jailbreaks are off-manifold" explanation.

1. Refusal-associated activation directions are behaviorally meaningful and can be manipulated causally.
2. Large random latent perturbations can sharply disrupt refusal.
3. The magnitude of local off-manifold displacement alone does not explain which perturbations cause failure.
4. The model's final state along behaviorally meaningful refusal-related structure may provide a more useful signal.
5. These failure-inducing internal states can now be used as adversarial examples for safety fine-tuning.

The broader goal is to understand **which internal representations support safe refusal behavior, how adversarial latent interventions disrupt those mechanisms, and whether training directly against those failure states can produce more robust models.**

---

## Repository Structure

```text
cross_model_pipeline/
├── configs/
│   └── models.py
├── pipeline/
│   ├── common.py
│   ├── 01_baseline.py
│   ├── 02_matched_refusal_direction.py
│   ├── 03_qwen_targeted_steering.py
│   ├── 04_qwen_rogue_scalpel.py
│   ├── 05_qwen_reference_bank.py
│   ├── 06_qwen_alignment_geometry.py
│   └── 07_qwen_refusal_axis_positions.py
└── test_model_loading.py
```

The repository also contains the original OLMo experiments used to develop and validate the pipeline.

---

## Limitations

The current behavioral refusal detector uses lightweight phrase matching and is intended as an experimental screening tool rather than a final safety evaluator.

"Coherent non-refusal" is also a screening category and should not be interpreted as equivalent to harmful compliance.

The local manifold analysis uses neighborhood PCA as an operational proxy for local activation geometry rather than assuming that the true representation manifold is locally linear.

Cross-model direction construction currently differs between OLMo and Qwen because of differences in refusal prevalence.

Future experiments will use stronger behavioral evaluation, additional model families, held-out jailbreak benchmarks, and latent adversarial fine-tuning.