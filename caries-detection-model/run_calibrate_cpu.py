import runpy, sys, torch
torch.cuda.is_available = lambda: False
sys.argv = ["pipeline.py", "calibrate",
            "--calib", r"@D:\Mahidol University\SP\caries-detection-model\outputs\peeked_cases.json",
            "--prefix", "best_pca_s42_f", "--n-folds", "5", "--recall-target", "0.85"]
runpy.run_path(r"D:\Mahidol University\SP\caries-detection-model\pipeline.py", run_name="__main__")
