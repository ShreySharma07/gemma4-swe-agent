"""Deterministic tests; synthetic trace data only, no shell/model execution."""
import unittest
from audit import kind, sequence, test_state, mutation_paths, path_class

class AuditTests(unittest.TestCase):
    def test_literal_source_path(self):
        cmd="python3 - <<'X'\nfrom pathlib import Path\np = Path('fastapi/routing.py')\np.write_text('x')\nX\n"
        self.assertEqual(mutation_paths('run_command',{'command':cmd}),['fastapi/routing.py'])
    def test_read_has_no_mutation_path(self):
        cmd="python3 - <<'X'\nfrom pathlib import Path\np=Path('fastapi/routing.py')\nprint(p.read_text())\nX\n"
        self.assertEqual(mutation_paths('run_command',{'command':cmd}),[])
    def test_scratch_distinct(self):
        self.assertEqual(path_class('reproduce_issue.py'),'scratch_or_test')
        self.assertEqual(path_class('/workspace/fastapi/routing.py'),'repository')
        self.assertEqual(path_class('unknown.py'),'unknown')
    def test_python_edit(self):
        self.assertEqual(kind('run_command',{'command':"python3 - <<'X'\np.write_text(s.replace(old,new))\nX"}),'mutation_intent')
    def test_read_not_edit(self):
        self.assertEqual(kind('run_command',{'command':"python3 - <<'X'\nprint(p.read_text())\nX"}),'read')
    def test_pytest_mention_not_execution(self):
        self.assertNotEqual(kind('run_command',{'command':"grep -n pytest tests/test_x.py"}),'test')
    def test_pytest_execution(self):
        self.assertEqual(kind('run_command',{'command':'cd /workspace && python3 -m pytest tests/test_x.py'}),'test')
    def test_missing_not_passed(self):
        self.assertEqual(test_state(None),'unknown')
        self.assertEqual(test_state({'status':'ok','stdout':''}),'unknown')
    def test_nested_failed(self):
        self.assertEqual(test_state({'status':'error','details':{'stdout':'1 failed'}}),'failed')
    def test_collection_distinct(self):
        self.assertEqual(test_state({'stderr':"ModuleNotFoundError: No module named 'x'"}),'environment_or_collection')
    def test_denied_distinct(self):
        self.assertEqual(test_state({'status':'error','error_type':'BudgetExceeded'}),'denied')
    def test_later_response_id(self):
        tr={'steps':[{'step_id':1,'tool_calls':[{'tool_call_id':'a','function_name':'run_command','arguments':{}}]}, {'step_id':2,'observation':{'results':[{'source_call_id':'a','content':'{"status":"ok"}'}]}}]}
        es,n=sequence(tr);self.assertEqual(n,0);self.assertEqual(es[0]['association'],'id')
    def test_ordered_batch(self):
        tr={'steps':[{'step_id':1,'tool_calls':[{'tool_call_id':x,'function_name':'run_command','arguments':{}} for x in ['a','b']], 'observation':{'content':'{"status":"ok"}','extra':{'tool_name':'run_command'}}},{'step_id':2,'observation':{'content':'{"status":"error","error_type":"BudgetExceeded"}','extra':{'tool_name':'run_command'}}}]}
        es,n=sequence(tr);self.assertEqual(n,0);self.assertEqual(es[1]['response']['error_type'],'BudgetExceeded');self.assertEqual(es[1]['association'],'ordered_same_tool')
    def test_explicit_wrong_id_never_guessed(self):
        tr={'steps':[{'step_id':1,'tool_calls':[{'tool_call_id':'a','function_name':'run_command','arguments':{}}], 'observation':{'results':[{'source_call_id':'b','content':'{}'}]}}]}
        es,n=sequence(tr);self.assertEqual(n,1);self.assertIsNone(es[0]['response'])

if __name__=='__main__':unittest.main()
