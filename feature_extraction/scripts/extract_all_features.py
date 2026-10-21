from pipeline import parser,run_branch

if __name__=='__main__':
    args=parser()
    run_branch('esm2',args)
    run_branch('prott5',args)
