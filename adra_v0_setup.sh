bash verl/scripts/install_vllm_sglang_mcore.sh # can ignore transformer engine installation error

pip install --extra-index-url https://pypi.nvidia.com --pre transformer-engine # re-install transformer engine

# --- VERL (local editable install, no deps — they're installed above) --------
cd verl
pip install --no-deps -e .
cd ..

# adra dependencies.
pip install trl==0.19.1 python-Levenshtein rank_bm25 math_verify antlr4-python3-runtime==4.13.2 scikit-learn evaluate seaborn nvidia-ml-py rich peft==0.17.1 sentence-transformers==5.2.0

# important! no dependency 
pip install --no-deps -e .


pip install omegaconf==2.3.0 # handle *Exception: Could not deserialize ATN with version 3 (expected 4).