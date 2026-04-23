# CIFAR-10 Accuracy Experiment Design

## Context

The current CIFAR-10 CNN pipeline run completed successfully through preprocessing,
training, evaluation, and model analysis, then stopped at promotion because the
candidate model did not meet the configured thresholds.

Observed baseline from the latest run:

- Validation accuracy: `0.7294`
- Validation weighted F1: `0.7315`
- Training config: 2 convolution layers, `10` epochs, batch size `32`,
  learning rate `0.001`
- Preprocessing config: native `32x32` size, normalization enabled, training
  augmentation disabled

The user wants a separate workspace and a small-first experiment sequence to see
whether limited tuning can improve accuracy before trying larger architectural
changes.

## Goal

Increase CIFAR-10 classification performance with low-risk changes first, while
keeping experiments easy to compare against the current pipeline result.

## Non-Goals

- Do not redesign the CNN architecture in the first pass.
- Do not change promotion thresholds during accuracy experiments.
- Do not change the CIFAR-10 import/split policy in the first pass.
- Do not treat a validation-only gain as sufficient evidence without also
  checking the test split.

## Constraints

- Work must happen in an isolated git worktree.
- The first pass should prefer config-level changes over training code changes.
- Experiments should remain attributable: each run should change one variable
  group at a time.
- Results should be compared against the current validation baseline for
  continuity, but test metrics should also be recorded to detect overfitting to
  the validation split.

## Experiment Strategy

The first experiment round will keep the current model architecture unchanged and
focus on the smallest adjustments with the highest likely return:

1. Reproduce the current baseline in the isolated workspace.
2. Increase training duration from `10` to `20` epochs.
3. Enable lightweight augmentation suitable for CIFAR-10:
   horizontal flips and small rotations.
4. If needed, combine the best settings from steps 2 and 3.

The sequence is intentionally narrow. If none of these changes produces a clear
improvement, the next round can consider training-loop improvements such as a
learning-rate scheduler, weight decay, or early stopping.

## Evaluation Policy

Each experiment run should record:

- Validation accuracy and weighted F1
- Test accuracy and weighted F1
- Training duration
- The exact config deltas from baseline

Validation metrics remain the primary comparison point because the current
pipeline already reports them. Test metrics are an additional guardrail and
should be used to reject changes that improve validation while degrading test
performance materially.

## Workspace Plan

Implementation will use a dedicated git worktree under the existing
`.worktrees/` directory. The experiment branch should be clearly named for the
task, for example `cifar10-accuracy-small-tuning`.

The worktree flow is:

1. Create the worktree and branch.
2. Confirm the local environment is usable there.
3. Run a baseline pipeline execution.
4. Apply one experiment change set at a time.
5. Save results in a compact comparison summary.

## Success Criteria

The first pass is successful if it produces a reproducible improvement over the
current validation baseline of `0.7294` without a meaningful regression on test
metrics.

Practical interpretation:

- Any small increase is useful as signal.
- A larger increase is preferable, but the main purpose of this pass is to learn
  which small knobs matter.
- If gains are negligible, that is still a valid outcome because it justifies
  moving to the next tier of changes.

## Risks

- Validation-only optimization may overstate improvement.
- Longer training may increase runtime without meaningful gains.
- Augmentation may help generalization, but if tuned poorly it can reduce fit on
  this small-image task.
- Existing pipeline behavior evaluates on validation by default, so experiment
  reporting must explicitly include test metrics to avoid ambiguous conclusions.

## Deliverables

The first implementation round should produce:

- An isolated worktree for the experiments
- A reproducible baseline run
- A small set of controlled experiment runs
- A concise results summary showing baseline vs each variant on validation and
  test metrics
- A recommendation on whether to continue with config tuning or move to
  training-loop/model changes

## Next Step

After this design is approved, the next artifact should be an implementation
plan that breaks the work into concrete steps for:

- worktree setup
- baseline reproduction
- experiment execution
- result collection
- comparison and recommendation
