# LuxAI-S1
LuxAI Season 1 Challange with Multi-agent Reinforcement Learning

LuxAI gym environment from glmcdona:
https://github.com/glmcdona/LuxPythonEnvGym
## Get started

Prerequisites:
* Python 3.8+
* [Poetry](https://python-poetry.org)

Install dependencies:
```
poetry install
```
Needs to be manually installed:
```
poetry shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip3 install gym=="0.22.0" stable-baselines3
```


Train agents against fixed opponent:
```
poetry run python -m luxai.luxrl.trainer.train 
```
Train agents via self-play:
```
poetry run python -m luxai.luxrl.ma.train_pfsp
```
Train agents with team-sprit:
```
poetry run python -m luxai.luxrl.ma.train_pfsp --team-sprit 0.2
```
