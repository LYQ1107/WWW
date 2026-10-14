"""One diagnostic process; original cache decisions and production code retained."""
import sys
from common import *
import gtr.geometry.association_trace as tracing
from gtr.geometry.cache_match_diagnostic import match_gates


class GateTrace(tracing.AssociationTrace):
    def adapt(self,outputs,instances,k):
        result=super().adapt(outputs,instances,k)
        self.records[-1]['cache_match_gates']=match_gates(self.model.geo_runtime.target_cache,instances[k])
        return result


def main():
    original=tracing.AssociationTrace
    tracing.AssociationTrace=GateTrace
    try:
        if '--trace-association' not in sys.argv:sys.argv.append('--trace-association')
        from run_online import main as run_online
        run_online()
    finally:tracing.AssociationTrace=original


if __name__=='__main__':main()
