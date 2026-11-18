from pathlib import Path
import sys,io,unittest,warnings
from importlib.metadata import version
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
from rise2.common import write_json,now,code_revision
def main():
 out=ROOT/'outputs/preparation';out.mkdir(parents=True,exist_ok=True)
 warnings.filterwarnings('error',category=UserWarning)
 log=io.StringIO();result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/'tests')))
 (out/'synthetic_tests.txt').write_text(log.getvalue(),encoding='utf-8')
 write_json(out/'verification.json',{'status':'PASS' if result.wasSuccessful() else 'FAIL','tests_run':result.testsRun,
  'failures':len(result.failures),'errors':len(result.errors),'verified_at':now(),'code_revision':code_revision(),
  'synthetic_training_and_evaluation_only':True,'real_model_training_performed':False,'real_model_inference_performed':False,
  'real_pape_fitting_performed':False,'real_final_test_evaluated':False,'python':sys.version,
  'packages':{p:version(p) for p in ['xgboost','numpy','scipy','scikit-learn','pandas','PyYAML','lightgbm','pyarrow']}})
 print(log.getvalue());print('PREPARATION REPORT: '+str(out/'verification.json'));return 0 if result.wasSuccessful() else 1

if __name__=='__main__':sys.exit(main())
