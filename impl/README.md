# Implementation 

Using Pytorch's Abstract Dataset class, I have created a class called HEARDS 
to simplify the process of loading the dataset. I assume that you have 
downloaded the Audiosnippets from [here](https://download.hz-ol.de/hear-ds-data/HEAR-DS/AudioSnippets/AudioSnippets-ITC-16kHz/).
After untarring the respective files, listing the files in the directory should
give you the following output:

```bash
$ ls -l /path/to/HEAR-DS Dataset/ 
-rw-r--r-- 1 user user 160 11 Nov 10:11 CocktailParty
-rw-r--r-- 1 user user 160 11 Nov 10:11 InTraffic
-rw-r--r-- 1 user user 160 11 Nov 10:11 InVehicle
-rw-r--r-- 1 user user 160 11 Nov 10:11 Music
-rw-r--r-- 1 user user 160 11 Nov 10:11 QuietIndoors
-rw-r--r-- 1 user user 160 11 Nov 10:11 ReverberantEnvironment
-rw-r--r-- 1 user user 160 11 Nov 10:11 WindTurbulence
etc.
```

To train and test the model, I have created scripts 'train.py' and 'test.py' 
respectively. Each of them instantiates the HEARDS class and one of the parameters 
is the root directory of the dataset. Change this accordingly in the scripts.
TODO: Use environment variables to set the root directory of the dataset.