# GPT2

In this project we repoduce the GPT architecture implementation and re-train it to match OpenAI 2019 results. 
This works for us as a distributed pretraining experiment.

The dataset chosen for pretraining is a 10B subset of the SlimPajama dataset (highly qualitative data). The training is performed on 3xA100 80GB. 

This project is inspired from: https://github.com/karpathy/build-nanogpt/tree/master
