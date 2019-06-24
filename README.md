OpenAI Gym environment of Threes! by Sirvo LLC. This code is based on the [Threes! AI](https://github.com/nneonneo/threes-ai) of nneonneo. You can get the game from here: http://asherv.com/threes/

It also includes code to interface with [Stable Baselines](https://github.com/hill-a/stable-baselines).

## Building

We first need to register the Threes! gym environment. To do this, execute 

    python setup.py install

Then run stable.py to train a model on this problem. 

## Running
### Python prerequisites

You will need Python 3.6, NumPy, Tensorflow, Openai Gym and Stable Baselines. I would recommend installing a virtual Anaconda environment 