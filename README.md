# Wilson's Morning Wake up Deep Playlist Generator

Create virtual environment
```bash
conda env create --file local_env.yml
```

Activate and deactivate environment
```bash
conda activate local_wmw
conda deactivate
```

Install latest pytorch for local and cpu only
```bash
conda install pytorch torchvision cpuonly -c pytorch
```

Install necessary packages (On your VM)
```bash
pip install -r gpu_requirements.txt
```
