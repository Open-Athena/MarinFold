It looks like the [ESMFold2 Atlas](http://biohub.ai/esm/protein/get-started) data is available under a permissive license (scroll down to "Download Atlas Data" in that link). 

We might be able to expand our training set a lot by including this data. I wonder:

What exactly is in this - only protein monomers?

How many new structural clusters would it give us?

What quality or other filters should we use?

Given our current eval set, is there possible leakage that we need to consider?

If there is a diverse ~10M-100M subset of proteins here that looks good it would be great to put it on huggingface similar to [afdb-24M](https://huggingface.co/datasets/timodonnell/afdb-24M) (unless it is already on huggingface somewhere?) Then we can try generating training documents from it.

