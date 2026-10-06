import sys, pathlib
sys.path.insert(0, "/tmp/claude-0/gsa/harness/official")
from adk_submission.compiler import compile_submission
from swegemma.config import build_submission_limits
from swegemma.models.registry import setup_gemma_model_registry
from swegemma.models.discovery import validate_single_declared_model
sub = pathlib.Path(sys.argv[1])
names = ["run_command","read_file","edit_file","write_file","get_status","submit_patch","get_code_neighbors","search_similar_code","get_code_subgraph"]
def mk(n):
    def f(**kw): return "{}"
    f.__name__ = n; f.__doc__ = n
    return f
tools = {n: mk(n) for n in names}
models = setup_gemma_model_registry(api_base="http://127.0.0.1:8000/v1")
print("declared model:", validate_single_declared_model(sub))
limits, gen = build_submission_limits()
agent = compile_submission(sub, tools, models, limits=limits, generation_constraints=gen)
def walk(a, d=0):
    print("  "*d + f"{type(a).__name__} {a.name}", "| tools:", [getattr(t,'name',getattr(t,'__name__',str(t))) for t in getattr(a,'tools',[]) or []], "| output_key:", getattr(a,'output_key',None), "| include_contents:", getattr(a,'include_contents',None))
    for s in getattr(a, 'sub_agents', []) or []: walk(s, d+1)
walk(agent)
