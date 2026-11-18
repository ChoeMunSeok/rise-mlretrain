"""Hash-verified, portable Validation handoff archive; no raw NPZ files."""
from pathlib import Path
import hashlib,json,zipfile
from .common import *
from .pipeline import verify_pape

def verify_bundle(path):
    with zipfile.ZipFile(path) as z:
        manifest=json.loads(z.read('bundle_manifest.json'))
        if set(z.namelist())!=set(manifest['files'])|{'bundle_manifest.json'}:raise ValueError('Bundle file list changed')
        for name,expected in manifest['files'].items():
            if name.startswith('/') or '..' in Path(name).parts:raise ValueError('Unsafe bundle path')
            if hashlib.sha256(z.read(name)).hexdigest()!=expected:raise ValueError('Bundle file changed: '+name)
        if manifest['target_role']!='validation' or manifest['partial_validation']:raise ValueError('Full Validation bundle required')
    return {'status':'PASS','files_checked':len(manifest['files']),'target_role':'validation'}

def make_bundle(cfg,result_run=None):
    result=latest(cfg,'pape_validation',result_run);verification=verify_pape(result)
    summary=read_json(result/'pape_summary.json')
    if summary['partial_validation']:raise ValueError('Partial Validation cannot be handed off as a complete experiment')
    export=Path(summary['source_export_run']);model=Path(read_json(export/'manifest.json')['model_run'])
    files={}
    for prefix,root in [('validation',result),('feature_exports',export),('model',model),('implementation/code',ROOT/'code'),('implementation/docs',ROOT/'docs')]:
        for p in sorted(root.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':files[prefix+'/'+p.relative_to(root).as_posix()]=p
    for name in ['README.md','run_validation.ps1','run.ps1','check.ps1','requirements.txt','requirements.lock.txt','config/study.yaml','scripts/run_pipeline.py','scripts/check_preparation.py']:
        files['implementation/'+name]=ROOT/name
    manifest={'schema_version':'apigraph_validation_handoff_bundle_v1','target_role':'validation',
      'source_period':summary['source_period'],'target_period':summary['target_period'],'partial_validation':False,
      'original_result_run':str(result),'created_at':now(),'synthetic_fixture':summary['synthetic_fixture'],
      'model_hash':summary['model_hash'],'code_revision':code_revision(),'validation_rows':summary['n_rows'],
      'dataset_hashes':DATA_HASHES,'logical_paths':{'results':'validation','export':'feature_exports','model':'model'},
      'operational_inputs':['validation/gain_inputs.parquet','validation/rccda_inputs.json'],
      'evaluation_only':['validation/pape_estimates.csv','validation/pape_summary.json','validation/confusion_matrix.csv','feature_exports/validation/labels_for_evaluation.parquet'],
      'note':'Original absolute paths remain as provenance. Use logical_paths for bundle content; raw NPZ files are supplied separately.',
      'files':{name:hash_file(p) for name,p in files.items()}}
    folder=local_path(cfg['paths']['outputs'])/'handoff';folder.mkdir(parents=True,exist_ok=True)
    path=folder/(result.name+'_'+__import__('datetime').datetime.now(__import__('datetime').timezone.utc).strftime('%H%M%S%f')+'.zip')
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for name,p in files.items():z.write(p,name)
        z.writestr('bundle_manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    check=verify_bundle(path)
    write_json(folder/'latest.json',{'bundle':str(path),'sha256':hash_file(path),'result_run':str(result),'verification':check})
    print('VALIDATION HANDOFF BUNDLE: '+str(path),flush=True)
    return path
