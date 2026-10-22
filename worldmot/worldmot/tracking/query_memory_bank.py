"""Global-ID memory separate from decoder slot indices."""

from __future__ import annotations


class QueryMemoryBank:
    def __init__(self, next_global_id: int = 1):
        self.next_global_id = int(next_global_id)
        self.archived: dict[int, dict] = {}

    def reset(self):
        self.next_global_id = 1
        self.archived.clear()

    def allocate(self, timestamp: int, observation_index: int) -> int:
        gid = self.next_global_id
        self.next_global_id += 1
        self.archived[gid] = {"birth_timestamp": int(timestamp), "last_observation": int(observation_index)}
        return gid

    def close(self, global_id: int, timestamp: int, reason: str = "max_missed"):
        if int(global_id) in self.archived:
            self.archived[int(global_id)]["death_timestamp"] = int(timestamp)
            self.archived[int(global_id)]["death_reason"] = str(reason)
