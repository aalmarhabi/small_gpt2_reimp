# Introduction
This is a side project to reimplement picoGPT that is a smaller version of Onpenai GPT model 2. The goal was to reimplement and understand the structure of GPT model. 

# Rsources
- Openai/gpt-2
- karpathy/nanoGPT
- **[jaymody/picoGPT](https://jaykmody.com/blog/gpt-from-scratch/?utm_source=tldrnewsletter#encoder)** (the most used resource - first check out jaymody repo and blog)
- arxiv.org - Attention Is All You Need

# Requirement
in addition to picoGPT requirements, jax, jaxlib, and simplejson. 
```
jax==0.4.4 
jaxlib==0.4.4 
regex==2017.4.5 
requests==2.27.1 
tqdm==4.64.0 
fire==0.5.0 
simplejson==3.18.3
tensorflow-macos==2.11.0
```
# Python version
``` Python 3.10.9 ```

# Run via CLI with default parameter
```
python gpt2.py "Test any text" 
```
# local Run
![](https://github.com/aalmarhabi/small_gpt2_reimp/blob/main/scr1.png)
