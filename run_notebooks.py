import subprocess

notebooks = [
    'ELM_10_SG_SVN.ipynb', 
    'ELM_10_SG1_MSC.ipynb', 
    'ELM_10_SG1_SVN.ipynb', 
    'XGBoost_10_SG_MSC.ipynb', 
    'XGBoost_10_SG_SVN.ipynb', 
    'XGBoost_10_SG1_MSC.ipynb', 
    'XGBoost_10_SG1_SVN.ipynb'
]

for nb in notebooks:
    print(f"Running {nb}...")
    result = subprocess.run([f"python3 -m nbconvert --to notebook --execute --inplace {nb}"], shell=True)
    if result.returncode == 0:
        print(f"Successfully completed {nb}")
    else:
        print(f"Failed to complete {nb}")
