# BAMBINO-LM: (Bilingual-)Human-Inspired Continual Pretraining of BabyLM

## Abstract
Children from bilingual backgrounds benefit from interactions with parents and teachers to re-acquire their heritage language. In this paper, we investigate how this insight from behavioral study can be incorporated into the learning of small-scale language models. We introduce BAMBINO-LM, a continual pretraining strategy for BabyLM that uses a novel combination of alternation and PPO-based perplexity reward induced from a parent Italian model. Upon evaluation on zero-shot classification tasks for English and Italian, BAMBINO-LM improves the Italian language capability of a BabyLM baseline. Our ablation analysis demonstrates that employing both the alternation strategy and PPO-based modeling is key to this effectiveness gain. We also show that, as a side effect, the proposed method leads to similar degradation in L1 effectiveness as human children would have had in an equivalent learning scenario.

## Training

### Prerequisites
Ensure you have `python3` and `pip` installed on your machine. Then, install the necessary dependencies via:
```
pip install -r requirements.txt
```
### Configuration
Parameters and configurations for training are located in [`config.json`](config.json). Adjust the settings as needed to customize your training process.

### Training
To start the training process, use the following command:
```
python3 train.py --config config.json
```

## Evaluation
Evaluations for our paper were conducted using EleutherAI's [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) on the [UINAUIL](https://github.com/valeriobasile/uinauil/tree/main) dataset. Please refer to the original repositories for usage guidelines.
