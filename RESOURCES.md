# Resources

## Code
Before this repo, we have been working in a few places (an early name for this project was LlamaFold):

This repo has early experiments in training LlamaFold models using huggingface transformers. It also has our data generation pipeline.
https://github.com/timodonnell/LlamaFold-experiments/tree/main

Later, we switched to marin and worked in this branch and experiment dir:
https://github.com/marin-community/marin/tree/protein-training-1b/experiments/protein

Now, with the creation of this repo, we are renaming the project to MarinFold and working out of here. We want all data curation, evals, and model training to happen in the MarinFold repo.

There are two other important repos to know about:

https://github.com/marin-community/marin-experiments
https://github.com/marin-community/marin

We want to follow the marin-experiments repo in that we will have independent little experiments that depend on marin. We also need to be aware of and watching the marin repo as it has the underlying implementations of many components we will use - iris, levanter, zephyr, etc. In particular we want to reuse the "experiment" infrastructure that Marin has set up, where we use issues to track experiments that are checked in as code.

## Datasets
So far we have been training entirely on AlphaFold Database (AFDB).

The “text” data (pre-tokenization) we are feeding these models is on huggingface [protein-docs](https://huggingface.co/datasets/timodonnell/protein-docs).
Note there are different “subsets” (which the huggingface preview doesn’t seem to render correctly) corresponding to different document layouts that I’ve tried over time. The different layouts are documented in the README.

The underlying AFDB data we’ve curated is also on huggingface [afdb-1.6M](https://huggingface.co/datasets/timodonnell/afdb-1.6M).

## Tokenizers
I was bad about saving the tokenizers for my early experiments, but the tokenizer for the “contacts-and-distances-v1” document type (currently the latest document type) is [here](https://huggingface.co/timodonnell/protein-docs-tokenizer).
I plan to keep adding to this as we make new document types.

**Checkpoints**

For models that seem particularly good or interesting (both pre-Marin and after switching to Marin models), I’ve been uploading them to huggingface [here](https://huggingface.co/timodonnell/LlamaFold-experiments/tree/main).

Model names should include the wandb run name so we can connect the two.

With the rename to MarinFold we should now put interesting checkpoints on [this](https://huggingface.co/buckets/timodonnell/MarinFold) bucket.

**Weights and biases**

After switching to Marin, runs are going here: [https://wandb.ai/timodonnell/marin](https://wandb.ai/timodonnell/marin)

Before Marin, I was sending runs to different projects for each experiment (names correspond to experiment dirs in the [LlamaFold-experiments](https://github.com/timodonnell/LlamaFold-experiments/tree/main) github repo): [exp4](https://wandb.ai/timodonnell/exp4?nw=nwusertimodonnell), [exp5](https://wandb.ai/timodonnell/exp5?nw=nwusertimodonnell), [exp6](https://wandb.ai/timodonnell/exp6?nw=nwusertimodonnell)

After the rename to MarinFold, let's put runs at https://wandb.ai/timodonnell/MarinFold (will need to create the project first time it is used).


