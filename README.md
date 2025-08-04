# The most hackable game engine (wannabe)

The purpose of this project is to provide a game engine that works in a 
simple, readable and transparent way and most importantly facilitates it for a user to 
easily implement and test their own ideas.

Implemented games: Chess, Hex.

### Installation

Generally should require just python>=3.8, project uses `uv` package manager.

Install python dependencies:
`uv sync`

##### For training your model

Project is prepared to train on Intel GPU with python==3.13, pytorch==2.7. For a different setup you're on your own.

### Basic Usage - run engine to find the best move in a position

Chess:
`PYTHONPATH=. python hackable_engine -G=chess -m -n="rn1qk2r/pp3ppp/2pb4/5b2/3Pp3/4PNB1/PP3PPP/R2QKB1R w KQkq - 0 10"`

Hex size 13:
`PYTHONPATH=. python hackable_engine -G=hex -S=13 -m -n=a4`

### Training

Multiple algorithms were used along the way, but the project settled with a custom implementation of DQN. 
Other algorithms are not facilitated to be run at this point. The following commands are just for the DQN training loop.

To train a model
`PYTHONPATH=. python hackable_engine/training/run.py -t -e=<ENVIRONMENT NAME>`

To load and retrain a model
`PYTHONPATH=. python hackable_engine/training/run.py -t -e=<ENVIRONMENT NAME> -v=<VERSION TO LOAD>`

To monitor training progress open tensorboard log in web browser
`tensorboard --logdir <PATH TO THE LOG DIRECTORY>`

### Development directions

- Train a state-of-the-art model for Hex
- Build a WASM version to run a Hex bot in a website
- Clean up and make the project more "hackable" as promised
- Make a lichess bot

##### Speeding up for python < 3.11

The best performance is provided by python 3.11. 

In case you use a different version there is a `compile.py` script that uses mypyc to compile shared libraries for python.

The compilation will only work if all types are correctly assigned in the code. 
This may require some additional work as with py3.11 I got lazy and didn't watch types carefully anymore.
