# The most hackable game engine (wannabe)

The purpose of this project is to provide a game engine that works in a 
simple, readable and transparent way and most importantly opens up to a user means to 
implement and test their own ideas.

Implemented games: Chess, Hex.

### Installation

Project tested with python 3.8 to 3.12 in debian and arch OS.

##### For running the engine

`pip install -r requirements.txt`

##### For training your model

`pip install -r hackable_engine/training/requirements.txt`

This will take more effort though, as it depends on your hardware. 

For instance as an Intel ArcA770 user I have intel torch extensions. For Nvidia GPU you may want CUDA-related stuff, etc.

### Basic Usage - run engine to find the best move in a position

Chess:
`PYTHONPATH=. python hackable_engine -G=chess -m -n="rn1qk2r/pp3ppp/2pb4/5b2/3Pp3/4PNB1/PP3PPP/R2QKB1R w KQkq - 0 10"`

Hex size 13:
`PYTHONPATH=. python hackable_engine -G=hex -S=13 -m -n=a4`

### Advanced Usage - implement your own criteria

WIP

### Training

The model files are saved into current working directory.

To train a model
`PYTHONPATH=. python hackable_engine/training/run.py -t -e=<ENVIRONMENT NAME>`

To retrain a model
`PYTHONPATH=. python hackable_engine/training/run.py -t -e=<ENVIRONMENT NAME> -v=<VERSION TO LOAD>`

To plot the training rewards over the training period
`PYTHONPATH=. python hackable_engine/training/run.py -pl -e=<ENVIRONMENT NAME>`

To view the tensorboard log, if used
`tensorboard --logdir <PATH TO THE LOG DIRECTORY>`

Add custom tensorboard logs in `hackable_engine/training/callbacks.py`.

### Development directions

- Build a WASM version to run a Hex bot in a website
- GUI with the board and sliders for criteria, option to check move suggestions by the engine with given criteria
- Make a lichess bot

##### Speeding up for python < 3.11

The best performance is provided by python 3.11. 

In case you use a different version there is a `compile.py` script that uses mypyc to compile shared libraries for python.

The compilation will only work if all types are correctly assigned in the code. 
This may require some additional work as with py3.11 I got lazy and didn't watch types carefully anymore.
