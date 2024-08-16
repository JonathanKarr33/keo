
# Trusted Knowledge Extraction for Maintenance and Manufacturing Intelligence

University of Notre Dame | Center for Research Computing | NSWC Crane Trusted AI  
**Authors**: Kate Mealey, Jonathan Karr, Danny Finch, Alyssa Riter, Priscila Moreira, Eli Phillips, Paul Brenner, Charles Vardeman

## Table of Contents
1. [Operations and Maintenance Intelligence (OMIN) Dataset](#omin-dataset)
2. [NLP Evaluated Tools](#nlp-tool-evaluation)
    - [Named Entity Recognition (NER)](#ner)
      - [spaCy EntityRecognizer](#spacy_ner)
      - [flair NER](#flair_ner)
      - [stanza NERProcessor](#stanza_ner)
      - [nltk ne_chunk](#nlkt_ner)
  
    - [Coreference Resolution (CR)](#cr)
      - [ASP](#asp_cr)
      - [coref_mt5](#corefmt5_cr)
      - [s2e-coref](#s2s_cr)
      - [neuralcoref](#neuralcoref_cr)
    
    - [Named Entity Linking (NEL)](#nel)
      - [BLINK](#blink_nel)
      - [spaCy EntityLinker](#spacy_nel)
      - [GENRE](#genre_nel)
      - [ReFinED](#refined_nel)
    
    - [Relation Extraction (RE)](#re)
      - [REBEL](#rebel_re)
      - [UniRel](#unirel_re)
      - [DeepStruct](#deepstruct_re)
      - [PL-Marker (SciERC)](#plmarker_re)
   
   
## Operations and Maintenance Intelligence (OMIN) Dataset <a name="omin-dataset"></a>

We present the **Operations and Maintenance Intelligence (OMIN) Dataset**, which is based on raw FAA Accident/Incident data. The OMIN dataset is curated specifically for knowledge extraction (KE) in operations and maintenance, featuring textual descriptions of maintenance incidents characterized by mentions of aircraft systems and domain-specific shorthand.

### Key Features:
- **Textual Descriptions**: Contains maintenance incident reports, including mentions of aircraft systems and domain-specific jargon.
- **Gold Standards**: Prepared for Named Entity Recognition (NER), Coreference Resolution (CR), and Named Entity Linking (NEL).
- **Structured Data**: Includes details such as aircraft specifics, failure codes, and incident dates, which can be used alongside the natural language text for integrated KE approaches.
- **Domain-Relevance**: While currently focused on aviation, the data has qualities common to other operation and maintenance records/logs, making it a valuable baseline for KE in these domains.

By publicizing this dataset, we aim to provide a resource for the maintenance and manufacturing community and encourage collaboration toward a robust, open-source KE dataset for this domain.

### Example:
![FAA Example](faa_example.png)


##  Survey knowledge extraction tools for maintenance data <a name="nlp-tool-evaluation"></a>

The survey of knowledge extraction tools for maintenance data includes tools for Named Entity Recognition (NER), Coreference Resolution (CR), Named Entity Linking (NEL), and Relation Extraction (RE). 
It provides an overview of each tool's methodology, base model, and relation set, where applicable. 

The survey also includes a comparison of the tools' outputs with the [Gold Standard Data](gold_standard/README.md), a subset of the Complete Set of FAA data created to evaluate the tools' correctness regarding NER, CR, and NEL tasks. There are two types of correctness evaluations: **quantitative** and **qualitative**. For more details on the correctness evaluations, please refer to the [Correctness Evaluation](evaluations/README.md).

KE tools at a glance:

| **[Named Entity Recognition (NER)](#ner)**  | **[Coreference Resolution (CR)](#cr)**   | **[Named Entity Linking (NEL)](#nel)**         | **[Relation Extraction (RE)](#re)**           |
|---------------------------------------------|------------------------------------------|------------------------------------------------|----------------------------------------------|
| [spaCy EntityRecognizer](#spacy_ner)        | [ASP](#asp_cr)                           | [BLINK](#blink_nel)                            | [REBEL](#rebel_re)                           |
| [flair NER](#flair_ner)                     | [coref_mt5](#corefmt5_cr)                | [spaCy EntityLinker](#spacy_nel)               | [UniRel](#unirel_re)                         |
| [stanza NERProcessor](#stanza_ner)          | [s2e-coref](#s2s_cr)                     | [GENRE](#genre_nel)                            | [DeepStruct](#deepstruct_re)                 |
| [nltk ne_chunk](#nlkt_ner)                  | [neuralcoref](#neuralcoref_cr)           | [ReFinED](#refined_nel)                        | [PL-Marker (SciERC)](#plmarker_re)           |


In the following sections, we provide a detailed explanation of the tools for each NLP task.

### Named Entity Recognition (NER) Tools<a name="ner"></a>

For each NER tool, we provide a list of the entity types it recognizes and indicate whether this tool is utilized in an NER subtask by other Coreference Resolution (CR), Named Entity Linking (NEL), or Relation Extraction (RE) tools.

The label sets used by our NER tools are detailed below:

| | **Entity Types** | **Tools Utilizing this Label Set** |
|---------------|------------------|------------------------------------|
| **OntoNotes** | Cardinal, Date, Event, Facility (FAC), Geo-Political Entity (GPE), Language, Law, Location (LOC), Money, Nationalities, Religious or Political Groups (NORP), Ordinal, Organization (ORG), Percent, Person, Product, Quantity, Time, Work of Art | spaCy, stanza, flair |
| **CoNLL-03**  | Person, Organization, Location, Miscellaneous Names | stanza, flair |
| **ACE-Phase-1** | Person, Organization, Location, Facility, Geo-Political Entity (GPE), Geographical-Social-Political Entity (GSP) | NLTK |
| **ACE-2005** | Person, Organization, Location, Facility, Geo-Political Entity (GPE), Vehicle (VEH), Weapon (WEA) | PL-Marker NER* |

\* see RE for PL-Marker

#### [spaCy EntityRecognizer]<a name="spacy_ner"></a>

[![Documentation](https://img.shields.io/badge/Documentation-View%20Here-blue?logo=readthedocs)](https://spacy.io/api/entityrecognizer)


EntityRecognizer identifies non-overlapping labeled spans of tokens using a transition-based algorithm. EntityRecognizer recognizes the 18 entity types in OntoNotes.

Model: en_core_web_sm, en_core_web_lg, and other spaCy models

_Used by:_ Our implementation of CoNLL-2012 format processing

#### [flair NER]<a name="flair_ner"></a>

[![Documentation](https://img.shields.io/badge/Documentation-View%20Here-blue?logo=readthedocs)](https://flairnlp.github.io/docs/intro)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/flairNLP/flair)


Flair ships several models which can be used for NER (or "tagging"), including their standard model, which recognizes 4 entity types and was trained for the CoNLL-03 task. They use transformer models, which they develop, and publish on Huggingface.

Model: flair/ner-english, flair/ner-english-ontonotes. Others available on [HuggingFace](https://huggingface.co/flair)

_Used by:_ BLINK

#### [stanza NERProcessor]<a name="stanza_ner"></a>

[![Documentation](https://img.shields.io/badge/Documentation-View%20Here-blue?logo=readthedocs)](https://stanfordnlp.github.io/stanza/ner.html)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/stanfordnlp/stanza/tree/main)


Stanza NERProcessor recognizes spans of mentions belonging to the 18 entity types found in the OntoNotes NER task.

Models: Default model is ontonotes-ww-multi_charlm, see [HuggingFace](https://huggingface.co/stanfordnlp/stanza-en/tree/main/models/ner) for other options. Model training is described in t\

Not used by any other tools in the pipeline.

#### [NLTK ne_chunk]<a name="nlkt_ner"></a>

[![Documentation](https://img.shields.io/badge/Documentation-View%20Here-blue?logo=readthedocs)](https://www.nltk.org/api/nltk.chunk.ne_chunk.html) 
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/nltk/nltk/tree/develop/nltk/chunk)


NLTK ne_chunk takes in a list of POS-tagged tokens as input and creates a parse tree where named entities and their labels are stored as subtrees. It recognizes the 6 entity types found in Phase 1 of the ACE project.

Not used by any other tools in the pipeline.

### Coreference Resolution (CR) tools<a name="cr"></a>

#### [ASP: Autoregressive Structured Prediction with Language Models]<a name="asp_cr"></a>

[![Paper](https://img.shields.io/badge/Paper-Read%20Now-brightgreen?logo=academia)](https://arxiv.org/pdf/2210.14698.pdf)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/lyutyuh/ASP)

The Autoregressive Structured Prediction (ASP) framework utilizes a conditional language model trained over structure-building actions, as opposed to strings, allowing the model to capture intra-structure dependencies more effectively and build pieces of the target structure step by step. It focuses on tasks such as named entity recognition, end-to-end relation extraction, and coreference resolution.

Base Model: T5

#### [coref_mt5: Coreference Resolution through a seq2seq Transition-Based System]<a name="corefmt5_cr"></a>

[![Paper](https://img.shields.io/badge/Paper-Read%20Now-brightgreen?logo=academia)](https://arxiv.org/pdf/2211.12142v1.pdf)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/google-research/google-research/tree/master/coref_mt5)

coref_mt5's methodology for coreference resolution uses a seq2seq approach, where a single sentence, along with prior context, is encoded as a string and fed into a model to predict coreference links. The system utilizes a transition-based approach, particularly the Link-Append system, which encodes prior coreference decisions in the input to the seq2seq model and predicts new coreference links as its output.

Base Model: mT5

#### [s2e-coref: Start-To-End Coreference Resolution]<a name="s2s_cr"></a>

[![Paper](https://img.shields.io/badge/Paper-Read%20Now-brightgreen?logo=academia)](https://aclanthology.org/2021.acl-short.3.pdf)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/yuvalkirstain/s2e-coref)

The s2e coreference resolution model introduces a lightweight approach that avoids constructing span representations. 
Instead, it uses the boundaries of spans to computes mention and antecedent scores, through a series of bilinear functions over their contextualized representations.

Base Model: longformer-large

#### [neuralcoref]<a name="neuralcoref_cr"></a>

[![Blog Post](https://img.shields.io/badge/Blog%20Post-Read%20Now-orange?logo=medium)](https://medium.com/huggingface/state-of-the-art-neural-coreference-resolution-for-chatbots-3302365dcf30)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/huggingface/neuralcoref)

Neuralcoref's methodology for coreference resolution uses the spaCy parser for mention-detection, and ranks possible mention-coreference pairs using a feedforward neural network developed by Clark and Manning, Stanford University (https://cs.stanford.edu/people/kevclark/resources/clark-manning-emnlp2016-deep.pdf).

Base Model: en_core_web_sm, en_core_web_lg, and other spaCy models

### Named Entity Linking (NEL) Tools<a name="nel"></a>

#### [BLINK: Scalable Zero-shot Entity Linking with Dense Entity Retrieval]<a name="blink_nel"></a>

[![Paper](https://img.shields.io/badge/Paper-Read%20Now-brightgreen?logo=academia)](https://arxiv.org/pdf/1911.03814.pdf)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/facebookresearch/BLINK)

BLINK introduces a two-stage zero-shot entity linking algorithm, utilizing a bi-encoder for dense entity retrieval and a cross-encoder for re-ranking. The bi-encoder independently embeds the mention context and entity descriptions in a dense space, while the cross-encoder concatenates the mention and entity text for more precise ranking. This approach demonstrates state-of-the-art performance on recent zero-shot benchmarks and traditional non-zero-shot evaluations, showcasing its effectiveness without the need for explicit entity embeddings or manually engineered mention tables.

Uses Flair for NER: https://github.com/flairNLP/flair \
Base Model: BERT

#### [spaCy EntityLinker]<a name="spacy_nel"></a>

[![Documentation](https://img.shields.io/badge/Documentation-View%20Here-blue?logo=readthedocs)](https://spacy.io/api/entitylinker)


spaCy EntityLinker is spaCy's NEL pipeline component. It uses the InMemoryLookupKB knowledge base to match mentions with external entities. InMemoryLookupKB contains Candidate components which store basic information about their entities, like frequency in text and possible aliases.

Model: en_core_web_sm, en_core_web_lg, and other spaCy models

#### [GENRE: Generative ENtity REtrieval]<a name="genre_nel"></a>

[![Paper](https://img.shields.io/badge/Paper-Read%20Now-brightgreen?logo=academia)](https://arxiv.org/pdf/2010.00904.pdf)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/facebookresearch/GENRE)

GENRE utilizes a sequence-to-sequence model to autoregressively generate textual entity identifiers. This approach allows GENRE to directly capture the relations between context and entity names, effectively cross-encoding both, and to efficiently compute the exact softmax for each output token without the need for negative data downsampling. Additionally, GENRE employs a constrained decoding strategy that forces each generated name to be in a predefined candidate set, ensuring that the generated output is a valid entity name.

Base Model: BART

#### [ReFinED: Representation and Fine-grained typing for Entity Disambiguation]<a name="refined_nel"></a>

[![Papers](https://img.shields.io/badge/Papers-Read%20Now-brightgreen?logo=academia)](https://arxiv.org/pdf/2207.04108.pdf), [![Papers](https://img.shields.io/badge/Papers-Read%20Now-brightgreen?logo=academia)](https://arxiv.org/pdf/2207.04106.pdf)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/amazon-science/ReFinED)

ReFinED is an efficient end-to-end entity linking model that utilizes fine-grained entity types and entity descriptions to perform mention detection, fine-grained entity typing, and entity disambiguation in a single forward pass. It targets a large catalog of entities, including zero-shot entities, and is capable of generalizing to large-scale knowledge bases such as Wikidata.

Base Model: RoBERTa.

Models: wikipedia_model, wikipedia_model_with_numbers, aida_model


## Relation Extraction (RE) Tools<a name="re"></a>

### [REBEL: Relation Extraction By End-to-end Language Generation]<a name="rebel_re"></a>
[![Paper](https://img.shields.io/badge/Paper-Read%20Now-brightgreen?logo=academia)](https://aclanthology.org/2021.findings-emnlp.204.pdf)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/Babelscape/rebel)

REBEL's methodology for relation extraction involves utilizing an autoregressive seq2seq model based on BART to express relation triplets as a sequence of text, simplifying the task of extracting triplets of relations between entities from raw text. This approach allows REBEL to perform end-to-end relation extraction for over 200 different relation types, and its flexibility enables it to adapt to new domains and datasets with minimal training time. Additionally, REBEL introduces a novel triplet linearization approach using special tokens, enabling the model to output relations in the form of triplets while minimizing the number of tokens that need to be decoded.

Base Model: BART-large

Relation set: Subset of 220 relations from Wikidata properties, found here: https://github.com/Babelscape/rebel/blob/main/data/relations_count.tsv

### [UniRel: Unified Representation and Interaction for Joint Relational Triple Extraction]<a name="unirel_re"></a>
[![Paper](https://img.shields.io/badge/Paper-Read%20Now-brightgreen?logo=academia)](https://arxiv.org/pdf/2211.09039.pdf)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/wtangdev/UniRel/tree/main)

UniRel's methodology for relation extraction involves unifying the representations of entities and relations by jointly encoding them within a concatenated natural language sequence. This approach fully exploits the contextualized correlations between entities and relations and leverages the semantic knowledge learned from pre-training. Additionally, UniRel proposes unified interactions to capture the interdependencies between entity-entity interactions and entity-relation interactions, achieved through the proposed Interaction Map built upon the off-the-shelf self-attention mechanism within any Transformer block.

Base Model: bert-base-cased

Relation set: 25 relations from the NYT dataset, found here: https://github.com/wtangdev/UniRel/blob/main/dataprocess/rel2text.py#L30

['/business/company/advisors','/business/company/founders','/business/company/industry','/business/company/major_shareholders','/business/company/place_founded','business/company_shareholder/major_shareholder_of','/business/person/company','/location/administrative_division/country','/location/country/administrative_divisions','location/country/capital','/location/location/contains',   '/location/neighborhood/neighborhood_of','/people/deceased_person/place_of_death','/people/ethnicity/geographic_distribution','/people/ethnicity/people','/people/person/children','/people/person/ethnicity','/people/person/nationality','/people/person/place_lived','/people/person/place_of_birth','/people/person/profession','/people/person/religion','/sports/sports_team/location','/sports/sports_team_location/teams']

### [DeepStruct]<a name="deepstruct_re"></a>
[![Paper](https://img.shields.io/badge/Paper-Read%20Now-brightgreen?logo=academia)](https://arxiv.org/pdf/2205.10475.pdf)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/wang-research-lab/deepstruct)

DEEPSTRUCT's methodology for relation extraction involves a sequence-to-sequence extraction approach using augmented natural languages. It formulates the task as two unit tasks: entity prediction to generate entities and relation prediction to generate relations, with a focus on generating triples for a wide set of structure prediction tasks in an end-to-end fashion. This approach decomposes structure prediction tasks into a collection of triple generation tasks, providing a unified representation for various structure prediction tasks without the need for introducing new data augmentation.

Base Model: GLM

| Dataset | Relation Set |
|----------|----------|
| CoNLL 04 | ['Work-For', 'Kill', 'Organization-Based-In', 'Live-In', 'Located-In'] |
| ADE | ['Adverse-Effect'] |
| NYT | ['/business/company/advisors','/business/company/founders','/business/company/industry','/business/company/major_shareholders','/business/company/place_founded','business/company_shareholder/major_shareholder_of','/business/person/company','/location/administrative_division/country','/location/country/administrative_divisions','location/country/capital','/location/location/contains',   '/location/neighborhood/neighborhood_of','/people/deceased_person/place_of_death','/people/ethnicity/geographic_distribution','/people/ethnicity/people','/people/person/children','/people/person/ethnicity','/people/person/nationality','/people/person/place_lived','/people/person/place_of_birth','/people/person/profession','/people/person/religion','/sports/sports_team/location','/sports/sports_team_location/teams'] |
| ACE05 | ['PER-SOC', 'ART', 'ORG-AFF', 'GEN-AFF', 'PHYS', 'PART-WHOLE'] |

### [PL-Marker: Packed Levitated Marker for Entity and Relation Extraction]<a name="plmarker_re"></a>
[![Paper](https://img.shields.io/badge/Paper-Read%20Now-brightgreen?logo=academia)](https://arxiv.org/pdf/2109.06067.pdf)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/thunlp/PL-Marker/tree/master?tab=readme-ov-file)

PL-Marker is a method for entity and relation extraction. The key innovation is the strategic use of levitated markers in the encoding phase to model the interrelation between spans and span pairs. Levitated markers are pairs of markers associated with a span, sharing the same position embedding with the start and end tokens of the corresponding span. They are used to classify multiple pairs of entities simultaneously and accelerate the inference process. The document also introduces neighborhood-oriented and subject-oriented packing strategies to consider the interrelation between spans and span pairs, enhancing the modeling of entity boundary information and the interrelation between same-subject span pairs.

PL-Marker uses different models for each dataset among ACE 2004, ACE2005, and SciERC:

| Dataset | Base Model | Relation Set |
|----------|----------|----------|
| ACE04 | albert-xxlarge-v1 or bert-base-uncased | ['PER-SOC', 'OTHER-AFF', 'ART', 'GPE-AFF', 'EMP-ORG', 'PHYS'] |
| ACE05 | albert-xxlarge-v1 or bert-base-uncased | ['PER-SOC', 'ART', 'ORG-AFF', 'GEN-AFF', 'PHYS', 'PART-WHOLE'] |
| SciERC | scibert_scivocab_uncased | ['PART-OF', 'USED-FOR', 'FEATURE-OF', 'CONJUNCTION', 'EVALUATE-FOR', 'HYPONYM-OF', 'COMPARE'] |

---

Please open an issue if you have any questions.

---
