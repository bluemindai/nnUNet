# Usage for mosaic:
1. #### You have to create directory 'trains' which will contains 3 mode directories (nnUNet_preprocessed, nnUNet_raw, nnUNet_results).
The folder structure should be like this:
```
└── trains
    ├── nnUNet_preprocessed
    ├── nnUNet_raw
    └── nnUNet_results
```
2. #### You have to set the env variables.
For example it should be like:
```bash
export nnUNet_raw="/bluemind/nnunet/trains/nnUNet_raw"

export nnUNet_preprocessed="/bluemind/nnunet/trains/nnUNet_preprocessed"

export nnUNet_results="/bluemind/nnunet/trains/nnUNet_results"
```

3. #### You have to put your dataset into nnUNet_raw directory and then create dataset.json.
Use [generate_json](scripts/generate_json.ipynb) for this.

4. #### Use scripts for training.
There are several scripts written by me (Roma) for datgaset  [preprocessing](scripts/preprocess_data.sh), [training](scripts/train.sh), [finetuning](scripts/finetune.sh) and [predicting](scripts/predict.py).

### READ NNUUNET [README](readme.md) FOR MORE INFORMATION
