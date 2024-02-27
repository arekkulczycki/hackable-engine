# The most hackable game engine

The purpose of this project is to provide a game engine that works in a 
simple, readable and transparent way and most importantly opens up to a user means to 
implement and test their own ideas.

Implemented games: Chess, Hex.

### Installation

Project tested with python 3.8 to 3.11 in debian and arch OS.

##### For running the engine

`pip install -r requirements.txt`

##### For training your model

`pip install -r hackable-engine/training/requirements.txt`

This will take more effort though, as it depends on your hardware. 

For instance as an Intel ArcA770 user I have intel torch extensions. For Nvidia GPU you may want CUDA-related stuff, etc.

### Basic Usage - run engine to find the best move in a position

Chess:
`PYTHONPATH=. python arek_chess -G=chess -p=1 -l=9 -m -n="rn1qk2r/pp3ppp/2pb4/5b2/3Pp3/4PNB1/PP3PPP/R2QKB1R w KQkq - 0 10"`

Hex size 13:
`PYTHONPATH=. python arek_chess -G=hex -S=13 -p=1 -l=9 -m -n=a4`

### Advanced Usage - implement your own criteria

### Training

To train a model
`PYTHONPATH=. python arek_chess/training/run.py -t -e=<ENVIRONMENT NAME>`

To retrain a model
`PYTHONPATH=. python arek_chess/training/run.py -t -e=<ENVIRONMENT NAME> -v=<VERSION TO LOAD>`

To plot the training rewards over the training period
`PYTHONPATH=. python arek_chess/training/run.py -pl -e=<ENVIRONMENT NAME>`

To view the tensorboard log if used
`tensorboard --logdir <PATH TO THE LOG DIRECTORY>`

### Development directions

- Build a WASM version to run a Hex bot in a website
- GUI with the board and sliders for criteria, option to check move suggestions by the engine with given criteria
- Make a lichess bot

##### Board handling speed up

Before running the engine there is one more step left in order to power-up the performance.  

In the board/mypy_chess.py file there is the python-chess library with injected code of board/board.py.  
Combination of those can then be compiled into a C shared object thanks to the mighty mypyc project.  

Temporarily rename board/mypy_chess.py to board/board.py (or copy contents into board.py),  
in the board directory run  
`mypyc board.py`  
Copy compiled <i>board.(...).so<i/> classes into board directory and revert board.py to the original state.  
The interpreter will now use the compiled board class which is 200-300% faster.  

##### Memory leaks

The engine uses shared memory to handle common data between processes.  
In case of failures you will have memory leaks and later get tons of warnings or errors with regard to not cleaned memory.

### Run engine


### Move search/choice criteria


### Additional notes
