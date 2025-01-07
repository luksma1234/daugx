# daugx
a data augmentation Library
___
# Notes
- Make sure the seed, which is generated randomly when no seed is provided is not generated using the standard
    generator. This has to be a new generator!
- Implement Tags as image_annotation modality
- Implement a functionality to bypass the dataloader (to implement an own dataloader)
- Online editor sends a minimum version requirement for each workflow. This requirement is then checked locally for match.
- Overthink the "data inflation" wording
___
## Getting started
    pip install daugx

## Features



#### Reproducibility of Results
daugx is build to be able to exactly reproduce full training runs. Each agent is initialized with a seed - if none is 
given as parameter, daugx will generate one. Same seeds will exactly reproduce the results. Since daugx enables you to 
share your workflow and seeds, everyone can reproduce your exact result. (Given the same starting conditions) This may 
be extremely useful in scientific publications.

#### Datasets and Modalities
To enable daugx to load your data, you must configure a Dataset. Datasets come with different modalities. A Dataset can 
have multiple modalities. One modality represents one type of input. Configuration of modalities differs. An 
augmentation workflow can consist of multiple datasets, therefore daugx is able to load multiple multimodal datasets. 
Multimodal datasets have to have at least one prime modality, which all additional modalities depend on. 
For an object detection dataset, the prime modality would be "image", while "image_annotation" would be an additional 
modality. This is because you can load an image without annotation, but no annotation without image. Prime- and 
Additional Modalities have to have an ID by which connects all modalities. There can be multiple prime modalities, in 
this case one is chosen by its share on the amount of data in all prime modalities. Additional modalities are optional 
by default. This means daugx tries to find matching data for each modality, if there is data missing for an additional 
modality, this modality will be ignored. Datasets can have multipliers, which alter the chance of loading data from this 
dataset. You can also set the background_percentage parameter, to ensure a percentage of loaded data to be background. 
Data is classified as background, if exactly one prime modality comes with no additional modalities. Therefore, the 
prime modality is standalone and classified as background.

#### Filters
Filters are optional and can be applied to any modality. The filter options differ by modality. There can only be one 
Filter per modality. Filters filter by metadata. Each modality type has different options for filtering. Filter on 
the prime modality defines weather the data is loaded at all. Filter on additional modalities are just filtering 
connected data.

#### Augmentations
Augmentations are used to slightly modify data, while preserving data integrity. The main purpose of daugx is to 
simplify the creation of complex Augmentation workflows, without losing track of things. Augmentations can be chained 
together in complex sequences. This is especially important when working with multiple datasets, which may need 
different augmentations to be applied. Augmentations are usually for one specific configuration of modalities, the 
editor will let you know if augmentations are not matching with dataset modalities. For each augmentation, an execution 
probability can be set. There can be multi-data Augmentations. In this case daugx will automatically make sure that 
sufficient data is loaded to satisfy the augmentation. This is done by randomly choosing and executing one of all 
incoming branches by their execution probability until enough data has reached the augmentation.

#### Visualization
tbd.
