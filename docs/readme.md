
### Goal is to get local mkdocs with api for metadata

conda create -y -n mmc-docs-env python=3.11

conda activate mmc-docs-env

pip install mkdocs mkdocs-material mkdocstrings[python]

mkdocs serve

open url in browser
   http://127.0.0.1:8000/