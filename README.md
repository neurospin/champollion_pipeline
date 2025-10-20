# Champollion pipeline - step-by-step tutorial

# 1. Get Started

You should first get pixi:

```
curl -fsSL https://pixi.sh/install.sh | bash
source ~/.bashrc
```

You then create a pixi environment (we place here this environment in the directory env_pixi):
```
mkdir env_pixi
cd env_pixi
pixi init -c conda-forge -c https://brainvisa.info/neuro-forge
pixi add anatomist soma-env=0.0 pip ipykernel
```

