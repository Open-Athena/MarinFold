---
title: Sampling protein contacts from a single sequence
slug: marinfold-single-sequence-contacts
author: Open Athena
date: 2026-09-23
published: false
math: false
toc: false
tags: [MarinFold, proteins]
summary: A figure-first look at contact prediction, structural accuracy, and useful sampling diversity.
---

Accurate folding remains harder when a protein has few sequence relatives. Improving here could also help protein design.

{{plotly: 01_predictors.json | title="Existing predictors across MSA depths" | mobile="01_predictors-mobile.json"}}

Helico does much better when we give it the answer: the ground-truth contact map, including non-contacts. This is an information upper bound.

{{plotly: 02_oracle.json | title="Helico with oracle contacts" | mobile="02_oracle-mobile.json"}}

Aside: confidence recognized the oracle above every matched random map on 20 preselected proteins. These are weak negatives.

{{plotly: 02b_confidence.json | title="Oracle versus matched random contacts" | mobile="02b_confidence-mobile.json"}}

This suggests generating candidate contact sets directly from sequence, trained on structures from existing predictors.

{{plotly: 03_method.json | title="MarinFold" | mobile="03_method-mobile.json"}}

Our current 1.47B-parameter model saw 248.584B raw tokens. Here is its contact accuracy, followed by the structures Helico produces from its top-L contacts.

{{plotly: 04_contacts.json | title="Contact R-precision" | mobile="04_contacts-mobile.json"}}

{{plotly: 05_folding.json | title="From predicted contacts to structures" | mobile="05_folding-mobile.json"}}

Low-depth proteins remain the challenge. Consensus R-precision is 0.561; oracle selection of an individual sample gives 0.528. The maps are distinct, but useful whole-map alternatives remain limited.

{{plotly: 06_sampling.json | title="Consensus versus oracle selection" | mobile="06_sampling-mobile.json"}}

Next: better candidate diversity, inference-time search, and post-training.

<!-- Placeholder prose. Copy figure captions from theme.py when integrating into
the site; each states the population, uncertainty, conditioning and oracle limits.
Use data/summary.csv and data/paired_deltas.csv for exact values. -->
