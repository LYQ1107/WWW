"""Additional real contract: raise AFTER a cloned native commit, then restore."""
import gmt_jev_verifier_native_closure as closure
from gtr.modeling.gmt_jev.native_plan_replay import RestorationProbe

def main():
    original_replay=closure.replay;original_save=closure.save
    def replay(*args,probe_exception=False,**kwargs):
        if not probe_exception:return original_replay(*args,**kwargs)
        model,fn,call_args,call_kwargs,*values=args
        def fail_after_commit(*a,**kw):
            fn(*a,**kw)
            raise RestorationProbe('Intentional exception after full cloned native commit')
        # The adapter's finally block must roll back mutated Gallery/Bank/RNG.
        return original_replay(model,fail_after_commit,call_args,call_kwargs,*values,**kwargs)
    def save(path,data):
        if path.name=='NATIVE_PLAN_CLOSURE_PREFIX_CONTRACT.json':
            path=path.with_name('NATIVE_PLAN_AFTER_MUTATION_EXCEPTION_CONTRACT.json')
            data=dict(data,exception_after_full_native_commit=True)
        elif path.name=='NATIVE_CLOSURE_PREFIX_VIDEO12.json':
            path=path.with_name('NATIVE_AFTER_MUTATION_EXCEPTION_PREFIX_VIDEO12.json')
        return original_save(path,data)
    closure.replay=replay;closure.save=save
    try:closure.main([12],64)
    finally:closure.replay=original_replay;closure.save=original_save
if __name__=='__main__':main()
