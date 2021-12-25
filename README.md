# The most hackable chess engine

The purpose of this project is to provide a chess engine that works in a 
simple, readable and transparent way and most importantly opens up to the users the means to 
implement and test their own ideas.

### Installation

Project developed with python3.8 in debian OS.

Make sure you have the ./linux_requirements.txt fulfilled.

Run  
`pip install -r requirements.txt`  
in your python3.8 virtual environment.

Moreover in the requirements.txt there is a commented instruction how to install larch-pickle, do it as well.

### Before you run

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


### Next steps in order to obtain performance boost


### Additional notes


### Key cool libs used:  
* python-chess  LINK 
* faster-fifo  LINK
* larch-pickle  LINK
* mypyc  LINK