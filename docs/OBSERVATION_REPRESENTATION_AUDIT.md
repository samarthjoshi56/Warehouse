# Observation Representation Audit

## Current state features

The DQN flattens these observation fields:

- Grid: per-slot occupancy, expected retrieval time, priority, group, top-of-stack flag, and height index.
- Incoming item: expected retrieval time, priority, a compressed compatibility-group value, and presence.
- Temporary storage: occupancy and min/max retrieval estimates.
- Stack summaries: height, free capacity, and compatibility.
- Schedule lookahead: future events in scenario C.
- Simulation time.

The action mask is applied during action selection and in the next-state DQN target.

## Identified limitation

The incoming item uses only `min(item.compatible_groups)` as its group feature. This is a lossy encoding when an item can use more than one group. Two items with different compatibility sets can share the same feature value. The action mask still prevents invalid placements, but the agent cannot directly distinguish the full compatibility sets from its state vector.

Stack group is also represented as a normalized scalar. This can imply an ordering or distance between group IDs, although group IDs are categorical.

## Proposed Observation V2 experiment

Do not change the production observation yet. Build a feature extractor for experiments:

1. Encode compatibility as a binary vector with one value per configured group.
2. Encode stack groups with one-hot vectors rather than a scalar.
3. Keep numeric attributes (height, retrieval time, priority, occupancy) separate from categorical features.
4. Keep action masking unchanged.
5. Compare V1 and V2 using the same architecture, training budget, train seeds, held-out seeds, and scenario.
6. Run at least 5 independent training seeds before drawing conclusions.

## Evaluation

Report per scenario:

- reshuffles per retrieved item
- total reshuffles
- failed and unplaced items
- retrieval delay
- episode return
- mean and standard deviation across episodes

Promote V2 only if it improves held-out performance consistently. Avoid changing rewards and observations in the same experiment, because that would make attribution unclear.
