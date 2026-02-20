bash verl/scripts/install_vllm_sglang_mcore.sh # can ignore transformer engine installation error

pip install --extra-index-url https://pypi.nvidia.com --pre transformer-engine # re-install transformer engine

# --- VERL (local editable install, no deps — they're installed above) --------
cd verl
pip install --no-deps -e .
cd ..

# adra dependencies.
pip install sentence-transformers==5.2.3

# important! no dependency 
pip install --no-deps -e .
