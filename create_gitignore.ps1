# Create a new .gitignore file
'' | Out-File .gitignore

# Add patterns to ignore common files and directories
Add-Content .gitignore "# Ignore virtual environment"
Add-Content .gitignore "venv/"

Add-Content .gitignore ""
Add-Content .gitignore "# Ignore Python bytecode files"
Add-Content .gitignore "__pycache__/"

Add-Content .gitignore ""
Add-Content .gitignore "# Ignore Jupyter notebooks checkpoints"
Add-Content .gitignore ".ipynb_checkpoints/"

Add-Content .gitignore ""
Add-Content .gitignore "# Ignore system files"
Add-Content .gitignore ".DS_Store"
Add-Content .gitignore "Thumbs.db"

Add-Content .gitignore ""
Add-Content .gitignore "# Ignore large files"
Add-Content .gitignore "*.dll"
Add-Content .gitignore "*.lib"
Add-Content .gitignore "*.pkl"

# Commit the changes
git add .gitignore
git commit -m "Add .gitignore to ignore common and large files"
git commit -m "Add .gitignore to ignore common files and directories"