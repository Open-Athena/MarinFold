# spec: contacts-v1

This document defines a spec for the *contacts-v1* document type. It is the input to a coding agent that will do the implementation.

## Example document

```
<contacts-v1>
<begin-sequence>
<pos-22> <phe>
<n-term> <pos-20>
<pos-21> <ala>
<c-term> <pos 22>
<pos-20> <ala>
<begin-structure>
<contact> <pos-20> <pos-21> 
<contact> <pos-22> <pos-21> 
<end>
```

## Details

We have two sections: a sequence section (starting with `<begin-sequence>`) and a structure section (starting with `<begin-structure>`).

### Sequence section
This consists of three kinds of statements.

`<POS-XXX> <RESIDUE>` indicates that position XXX is the given amino acid. We have indexing tokens `<pos-0000>` through `<pos-1999>` (2000 indices total).

`<n-term>` `<pos-XXX>` indicates that position XXX is the N-terminus of a protein chain

`<c-term>` `<pos-XXX>` indicates that position XXX is the C-terminus of a protein chain.

The statements of the sequence section are given in random order. We define the amino acid for all residues exactly once. We define the N- and C-termini for each protein chain.

### Residue indexing
We support structures with up to 2000 residues

Rather than numbering residues in a protein as e.g. 0 to 1999, each time we generate a document we pick a random number n in [0, 2000) to be the n-terminal residue. We start indexing from this residue. Residue indices "wrap around", so the residue after `<pos-1999>` is `<pos-0>`. The motivation here is that we want the model to be experienced in using all residue indices. Since most proteins are way less than 2000 residues, if we always started the protein chain off at `<pos-0>` the model would only rarely see the higher value indices.

In the future we will support multiple protein chains, and we will just have multiple <n-term> and <c-term> statements for these. They will need to be spaced out enough to not overlap. For example we might have one protein that starts at index 1800 and continues until residue 100, and another that starts at residue 300 and continues until 800.

### Structure section
The structure section consists of statements of the form `<contact>` `<pos-XXX>` `<pos-YYY>`, which indicates that residues at index XXX and YYY are in contact.

Contacts are defined as contact degree > 0 where contact degree is implemented in [pyconfind](https://github.com/timodonnell/pyconfind). We run pyconfind in `native_only=True` mode, i.e. only consider the actual given amino acid at each position rather than all other possibilities.

Contacts are listed in the structure section from highest contact degree (strongest contact) to lowest. Note that the contact matrix is symmetric. We randomize the order order that the contact pair is given in: if there is a contact between XXX and YYY, with 50% probability we output `<contact> <pos-XXX> <pos-YYY>` and the other half of the time we output `<contact> <pos-YYY> <pos-XXX>`. Each contact is only specified once, i.e. once we see `<contact> <pos-XXX> <pos-YYY>` we will never see `<contact> <pos-XXX> <pos-YYY>` or `<contact> <pos-YYY> <pos-XXX>` later on.

### Document length
Our max document length 8192 tokens. If there are more contacts than can fit in that, we truncate.

### Additional tokens
For the vocab, also include as additional tokens all the tokens in the contacts-and-distances-v1 [vocab](https://github.com/Open-Athena/MarinFold/blob/main/marinfold/marinfold/document_structures/contacts_and_distances_v1/vocab.py). We may fine tune on documents like that later. Also include this additional token: `<think>`
