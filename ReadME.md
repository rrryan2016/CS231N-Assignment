# CS231N Assignment 
Most of the code online run well in a python 2 environment

## Assignment 2 
### Version Problem: The dfferences bewteen python 2 and 3 need to be noticed 
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


### Non-version Problem
* range()
> the feedback of *range()* is range object, if you wanna a list. Transfer the `a = range(0,N)` into `a = list(range(0,N))`
* Module *scipy.misc*
> You may come across error report concerning to the module *scipy.misc*. The most possible solution is to install *PIL*, by `pip install -U PIL`. The specific reason can take this website as reference.  https://stackoverflow.com/questions/15345790/scipy-misc-module-has-no-attribute-imread#