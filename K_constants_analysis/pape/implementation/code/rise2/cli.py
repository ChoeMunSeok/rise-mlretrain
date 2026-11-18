import argparse
from .common import *
from .data import audit
from .model import train_stage,export_stage
from .pipeline import pape_stage,verify_pape
from .bundle import make_bundle,verify_bundle

def run_validation(cfg):
    model=train_stage(cfg)
    export=export_stage(cfg,model_run=model)
    result=pape_stage(cfg,export_run=export)
    verify_pape(result)
    return make_bundle(cfg,result)

def main():
    p=argparse.ArgumentParser(description='Fresh 2012 model / Source 2013-2014 / Validation 2015 only')
    p.add_argument('stage',choices=['plan','audit','train','export','pape-validation','verify','bundle','verify-bundle','run-validation'])
    p.add_argument('--config',default='config/study.yaml');p.add_argument('--model-run');p.add_argument('--export-run')
    p.add_argument('--validation-run');p.add_argument('--bundle-path');p.add_argument('--max-chunks',type=int,default=0)
    a=p.parse_args();cfg=load_config(a.config)
    if a.max_chunks and a.stage!='pape-validation':p.error('--max-chunks is only supported for Validation pilots')
    if a.stage=='plan':
        view={k:cfg[k] for k in ['protocol_version','study_name','training','execution','split','chunks']}
        view['pape']={'n_features':len(cfg['pape']['features']),'seeds':cfg['pape']['seeds'],'weightings':cfg['pape']['weightings']}
        print(__import__('yaml').safe_dump(view,sort_keys=False))
    elif a.stage=='audit':
        out=local_path(cfg['paths']['outputs'])/'data_audit';out.mkdir(parents=True,exist_ok=True)
        write_json(out/'audit.json',audit(cfg));print(out/'audit.json')
    elif a.stage=='train':train_stage(cfg)
    elif a.stage=='export':export_stage(cfg,model_run=a.model_run)
    elif a.stage=='pape-validation':pape_stage(cfg,export_run=a.export_run,max_chunks=a.max_chunks)
    elif a.stage=='verify':
        out=latest(cfg,'pape_validation',a.validation_run);result=verify_pape(out);write_json(out/'verification.json',result);print(result)
    elif a.stage=='bundle':make_bundle(cfg,a.validation_run)
    elif a.stage=='verify-bundle':
        path=a.bundle_path or read_json(local_path(cfg['paths']['outputs'])/'handoff/latest.json')['bundle'];print(verify_bundle(path))
    else:run_validation(cfg)

if __name__=='__main__':main()
