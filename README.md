## Forge2 Spaces implementation of Switti ##
New Forge only.

derived from https://huggingface.co/spaces/dbaranchuk/Switti
see also https://github.com/yandex-research/switti

* *512 model*: 9.22GB
* *1024 model*: 9.53GB
* *Text encoders*: 1.59GB + 9.46GB
* *VAE*: 0.42GB

approximate reasonable minimum requirements: 8GB VRAM, 16GB RAM

>[!NOTE]
>Install via *Extensions* tab; *Install from URL* sub-tab; use URL of this repo
>Modified to run in bfloat16. Includes all repo files, 'Install' in Spaces tab will not be necessary.
>First run of a model loads the models, so can be significantly slower.
>Downloads on demand.

>[!NOTE]
>Saves to Forge `outputs` directory, using filename format `Switti_{datetime}.png`.

Example, using 512px model:
![](Switti_20241230230221.png "path through forest, lake, mountain background")