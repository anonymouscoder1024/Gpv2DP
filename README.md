# Gpv2DP
Software Defect Prediction via Code Grayscale Pixel Visualization with Fusion Attention(Gpv2DP).

We proposed a novel Gpv2DP approach, which simultaneously considers code visualization comprehension and fusion attention mechanism to extract the code feature for software defect prediction.


Prepare for dataset
=================
- `Put the source code of the experiment project in folder ./data/archives. Available via github or other websites.`
- `Generate the code images corresponding to the java files. Please run make_txt.py, code_vis.py, and makeInstance_txt.py sequentially.`
- `The path structure of the prepared data images is as follows:`

```
-data
  -archives
  -csvs
  -img
    - Project 1
      -file 1.png
      -file 2.png
      ...
    - Project 2
      -file 1.png
      -file 2.png
      ...
  -txt
```


Build running environment
=================
- `Install required packages.`

```
pip install -r requirements.txt
```

- `Install accimage (support Windows, Linux and macOS). Official link: https://github.com/pytorch/accimage`

```
$ conda install -c conda-forge accimage
```

Train and test
=================
- `To train and test, simply run train.py.`
```
python train.py
```

Config batch training
===============

- `Modify and execute the run.sh.`

```
sh run.sh
```

Check experimental result
===============
- `Please check the folder ./temp/result/.`


Contacts
===============
- `Due to anonymity requested, we will publish contact details in the future.`
