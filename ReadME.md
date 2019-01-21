# CS231N Assignment 
This is my job in CS231n (2016 Winter Jan-Mar) assignment!

Most of the code online run well in a python 2 environment and the original .ipynb file is not that convenient to debug or edit, this project can run well in Python 3.7!

My developing environment is Windows 10 + Pychram + Python 3.7.

Please properly comment or uncomment for each function. Each code block start with a 'start' hint and end with a 'end' hint. For example,
`#### Start: Test softmax and SVM in layers.py ####` & `#### End: Test softmax and SVM in layers.py ####`

Recommend you to read the official .ipynb file when using my codes. 

For dataset download:

* Assignment 2 

According to the instruction in official document, you can run `get_datasets.sh` in \cs231n\datasets or,

Open `get_datasets.sh` in a plain text editing tool, and follow its instruction so as to download. 

* Assignment 3 

According to the instruction in official document, you can run `get_tiny_imagenet_a.sh` `get_pretrained_model.sh` `get_coco_captioning.sh` in \cs231n\datasets or, 

Open these 3 files in a plain text editing tool (for example, Sublime), copy the website and start downloading dataset,then unzip zipped package(all as the instruction in .sh files). 

If the *pretrained_model* file end with *.txt*, you can directly rename it as `pretrained_model.h5`

## Assignment 2 

### Designed CNN by my Own in /classierfiers/customedCNN.py at the last

The structure is, which is the best one among those I tried,

**INPUT --> [CONV --> RELU --> POOL]*2 --> [CONV --> RELU] --> FC/OUT**

In the part, *Train the Net*, its result is,
`(Epoch 1 / 1) train acc: 0.470000; val_acc: 0.483000`

I also tried： 

 **INPUT --> [CONV --> RELU --> POOL]*2 --> [CONV --> RELU] --> FC --> ReLU --> FC/OUT **
 
 **INPUT --> [CONV --> ReLU]\*2 --> [FC --> ReLU]\*2 --> FC/OUT**
 
 **INPUT --> [CONV --> ReLU]\*2 --> POOL --> [FC --> ReLU]\*2 --> FC/OUT**

These networks perform less effcient in trainning, you may need to increase the `num_epochs` in `Solver()`.

### Some modification of the original codes 

#### Version Problem: The dfferences bewteen python 2 and 3 need to be noticed here

* datadict = pickle.load(f, encoding='bytes')
> The default decode method of *pickle.load()* is encodeing="ASCII". If the file to load is not in the saving form of ASCII, we need to choose a parameter in encoding. 
> 
> encoding='bytes' means load 8-bits string in the form of bytes.
 
* datadict[b'labels'] /  datadict[b'data']
> The original Y = datadict['labels'] X = datadict['data'] need to do this transfer. In 3, str is unicode(default); in 2, str is bytes(default), b'' mindicates byte.
> 
> No meaning of b'' in 2, only to be compatible with 3. 
> 
> Here to add b is to let code of 2 to suit code in 3 

* tranfers *iteritems* into *items*
> just a difference between 2 and 3. 

* Sign of division */* & *//*
> In python 2, / only keep the integer part, while the decimal part is abandoned. So the result is *int* . 
> 
> In python 3,  the result of */* is *float*. You can use *//* to get *int* result. 
> 
> You may come across it when you use range(), the parameter in bracket may be the result of a division. 

* Roughly, *xrange()* in Python2 was renamed to *range()* in Python3


#### Non-version Problem
* range()
> the feedback of *range()* is range object, if you wanna a list. Transfer the `a = range(0,N)` into `a = list(range(0,N))`

* Module *scipy.misc*
> You may come across error report concerning to the module *scipy.misc*. The most possible solution is to install *PIL*, by `pip install -U PIL`. The specific reason can take this website as reference.  https://stackoverflow.com/questions/15345790/scipy-misc-module-has-no-attribute-imread#



## Assignment 3 
Changes of teachers' original codes are similar to that in assignment 2. 

I applied *BLEU_score* as an evaluation for a good captioning model in LSTM_Captioning.