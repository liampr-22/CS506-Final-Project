Windows PowerShell usage (conda)

1. Create the environment from the YAML (run from project root):

   conda env create -f envs\environment-cs506.yml

2. Activate the environment:

   conda activate cs506

3. Start JupyterLab:

   jupyter lab

Notes:
- The YAML prefers conda-forge channel for geospatial and newer packages.
- If you already have an environment named `cs506`, either remove it (`conda env remove -n cs506`) or edit the `name:` in the YAML before creating.
- To export an updated environment after installing packages:

   conda env export -n cs506 --no-builds > envs\environment-cs506.yml
