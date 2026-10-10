"""No alias: retain opaque native IDs and expose observable competing support."""


class IdentityFragmentController:
    alias_enabled = False

    def aliases(self): return {}

    def verify(self, committed_ids):
        assert len(committed_ids) == len(set(committed_ids)), 'same-camera duplicate Global ID'
        return True
