"""Retrieval, blocker relocation, and deterministic restoration."""
from .exceptions import ItemNotFoundError, RestoreFailure


class RetrievalEngine:
    def __init__(self, stacks, temp_storage, metrics, safe_temp_threshold=6):
        self.stacks = stacks
        self.temp_storage = temp_storage
        self.metrics = metrics
        self.safe_temp_threshold = safe_temp_threshold

    @staticmethod
    def urgency_key(item):
        return item.urgency_key

    def blocker_count(self, item_id: str) -> int:
        for stack in self.stacks:
            try:
                pos = stack.position_of(item_id)
                return stack.height - pos - 1
            except ValueError:
                continue
        return 0

    def locate(self, item_id: str):
        for stack in self.stacks:
            try:
                return stack, stack.position_of(item_id)
            except ValueError:
                pass
        return None, None

    def retrieve(self, item, now: int):
        """Execute one retrieval atomically. Returns (reshuffles, elapsed_ticks)."""
        if item.location == "temp":
            self.temp_storage.remove(item.id)
            item.location, item.status = "out", "retrieved"
            self.metrics.data.retrieved += 1
            delay = max(0, now - item.expected_retrieval_time) if item.expected_retrieval_time is not None else 0
            self.metrics.data.total_retrieval_delay += delay
            return 0, 1

        stack, pos = self.locate(item.id)
        if stack is None:
            raise ItemNotFoundError(f"Retrieval target {item.id} is not stored")
        blockers = stack.height - pos - 1
        if self.temp_storage.free_slots < blockers:
            raise RuntimeError("Insufficient temporary storage")

        # B2: remove blockers top-first; each blocker moves only to temp storage.
        moved = []
        for _ in range(blockers):
            blocker = stack.pop()
            self.temp_storage.add(blocker)
            moved.append(blocker)
            self.metrics.data.reshuffles += 1

        target = stack.pop()
        target.location, target.status = "out", "retrieved"
        self.metrics.data.retrieved += 1
        delay = max(0, now + 2 * blockers + 1 - item.expected_retrieval_time) if item.expected_retrieval_time is not None else 0
        self.metrics.data.total_retrieval_delay += delay

        # Restoration is performed immediately after extraction.
        restored = self.restore()
        self.metrics.data.restores += restored
        return blockers, 2 * blockers + 1

    def restore(self):
        """Restore temp items in reverse urgency order with explicit occupancy accounting."""
        restored_ids = []
        pending = sorted(self.temp_storage.items, key=self.urgency_key, reverse=True)
        for item in pending:
            valid = [s for s in self.stacks if s.free_slots > 0 and item.is_compatible(s.group)]
            if not valid:
                continue

            compatible = []
            for stack in valid:
                top = stack.top
                if top is None or self.urgency_key(item) <= self.urgency_key(top):
                    compatible.append(stack)
            if compatible:
                destination = min(compatible, key=lambda s: (s.height, s.index))
                self.temp_storage.remove(item.id)
                destination.push(item)
                restored_ids.append(item.id)
                continue

            # Tier 2: retain staged items while occupancy is at/below the safe threshold.
            if self.temp_storage.count <= self.safe_temp_threshold:
                continue

            # Tier 3: temp pressure requires the least damaging non-LIFO placement.
            def inversion_gap(stack):
                top = stack.top
                if top is None:
                    gap = 0.0
                else:
                    a, b = self.urgency_key(item), self.urgency_key(top)
                    # Positive gap means this placement puts a later-retrieved item on top.
                    if a[0] == float("inf") or b[0] == float("inf"):
                        gap = 0.0
                    else:
                        gap = max(0.0, float(a[0] - b[0]))
                return (gap, stack.height, stack.index)

            destination = min(valid, key=inversion_gap)
            self.temp_storage.remove(item.id)
            destination.push(item)
            restored_ids.append(item.id)
        return len(restored_ids)
