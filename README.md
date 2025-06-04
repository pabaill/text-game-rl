# text-game-rl

Course project for CS 224R. Learning how to finetune LLMs to recognize relationships between objects in the world using RL.

## Getting started: Jericho

1. Create a new python environment (conda create -n cs224rproject)
2. Activate the python environment (conda activate cs224rproject)
3. Install the requirements (pip install -r requirements.txt)
4. Clone jericho into the same folder as the repo (git clone https://github.com/microsoft/jericho.git)
5. Install jericho (pip install .) (You may need to run "sudo apt update" then "sudo apt install build-essential" to build the jericho module)
6. Then install spacy dependencies (python3 -m spacy download en_core_web_sm)
7. Get games on your phone (wget https://github.com/BYU-PCCL/z-machine-games/archive/master.zip)
8. Unzip the games (unzip master.zip)
9. Read the quickstart guide [here](https://jericho-py.readthedocs.io/en/latest/tutorial_quick.html)

## Soft Actor Critc

We use online training in Jericho games for a Soft Actor Critic network.
SAC is an off-policy algorithm that uses entropy regularization. It is designed to maximize the tradeoff between expected return and entropy, meaning that it encourages both more exploration with increased entropy and high exploitation with expected return.
For more information, read OpenAI's docs: https://spinningup.openai.com/en/latest/algorithms/sac.html (their module not used, but provides good information)
