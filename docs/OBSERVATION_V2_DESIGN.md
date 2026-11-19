# Observation V2 Design

## Goal
Improve the DQN observation without changing warehouse dynamics, action legality, or reward.

## V1 limitations
- Incoming-item compatibility is compressed to the minimum compatible group ID. This loses information when an item supports multiple groups.
- Stack group is encoded as one numeric scalar. The network can interpret group IDs as ordered values, although group IDs are categories.
- The action mask shows which stacks are currently legal. It does not describe the full compatibility set independently of stack availability.

## V2 encoding
V2 is opt-in. V1 remains the default and is unchanged.

### Per-slot grid
Replace the scalar stack-group channel with a one-hot group encoding.
Keep occupancy, normalized expected retrieval time, normalized priority, top-of-stack indicator, and normalized slot height.
Shape: `(num_stacks, stack_height, 5 + num_groups)`.

### Incoming item
Replace the compressed compatibility scalar with a binary compatibility vector.
Keep normalized expected retrieval time, normalized priority, and item-present indicator.
Shape: `(3 + num_groups,)`.

### Other fields
Keep temporary storage, stack summaries, schedule lookahead, action mask, and simulation time unchanged.

## Experiment protocol
- Keep simulator dynamics and reward identical.
- Compare V1 and V2 with identical training seeds, episode counts, configuration, and held-out evaluation seeds.
- Report retrieval count, reshuffles, failed/unplaced items, delay, reward, and runtime.
- Use at least five training seeds before making an improvement claim.
- Do not promote V2 based on training reward alone.
