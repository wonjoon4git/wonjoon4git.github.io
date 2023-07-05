---
layout: single
classes: wide
title:  "Markov NLP"
categories: 
  - Machine Learning
tag: [NLP]
toc: true 
author_profile: false 
sidebar:
    nav: "counts"
---
# Markov Models of Natural Language Processing


## Language Models

Many of you may have encountered the output of machine learning models which, when "seeded" with a small amount of text, produce a larger corpus of text which is expected to be similar or relevant to the seed text. For example, there's been a lot of buzz about the new Chat GPT, and, yes, impressive.

We are not going to program a complicated deep learning model, but we will construct a much simpler language model that performs a similar task. Using tools like iteration and dictionaries, we will create a family of **Markov language models** for generating text. For the purposes of this assignment, an $n$-th order Markov model is a function that constructs a string of text one letter at a time, using only knowledge of the most recent $n$ letters. You can think of it as a writer with a "memory" of $n$ letters. 

## Data

Our training text for this exercise comes from the first 10 chapters of Jane Austen's novel *Emma*, which I retrieved from the archives at [Project Gutenberg](https://www.gutenberg.org/files/158/158-h/158-h.htm#link2H_4_0001). Intuitively, we are going to write a program that "writes like Jane Austen," albeit in a very limited sense. 


```python
len(s)
```




    158809



```python

s = ""
# Original text rxcluded due to length
# Basically the whole chapter of a novel
```



## Exercise 1
 
Write a function called `count_characters()` that counts the number of times each character appears in a user-supplied string `s`. The function should loop over each element of the string, and sequentually update a `dict` whose keys are characters and whose values are the number of occurrences seen so far. You may know of other ways to achieve the same result. However, we shall use the loop approach, since this will generalize to the next exercise. 

*Note: while the construct `for letter in s:` will work for this exercise, it will not generalize to the next one. Use `for i in range(len(s)):` instead.* 

### Example usage: 

```python
count_characters("tortoise")
{'t' : 2, 'o' : 2, 'r' : 1, 'i' : 1, 's' : 1, 'e' : 1}
```


###  Solution


```python
# write count_characters() here 
def count_characters(s):
    """
    find unique characters and its occurances within the given string.
    
    Parameters
    ----------
    s: string provided by the user, exploited in forloop to update dictionary. 
    
    Return 
    ----------
    D: Dictionary containing unique character keys and its occurances as value.
    """
    D={}
    for i in range(len(s)):
    # implementation of range(len(s)) for future use as mentioned
        substr=s[i]
        
        if substr in D:
        # take the count of substring if it alreay exists in dictionary
            count = D[substr]
        else:
        # if substring is not in dictionary, set a new count
            count = 0
            
        # update the dictionary with incremented count
        D.update({substr:count+1})
        
    return D

count_characters("tortoise")
```




    {'t': 2, 'o': 2, 'r': 1, 'i': 1, 's': 1, 'e': 1}



## Exercise 2

An `n`-*gram* is a sequence of `n` letters. For example, `bol` and `old` are the two 3-grams that occur in the string `bold`. 

Write a function called `count_ngrams()` that counts the number of times each `n`-gram occurs in a string, with `n` specified by the user and with default value `n = 1`. Only a small modification to `count_characters()` will enable to do this. 

### Example usage: 

```python
count_ngrams("tortoise", n = 2)
```
```
{'to': 2, 'or': 1, 'rt': 1, 'oi': 1, 'is': 1, 'se': 1} # output
```

###  Solution


```python
# write count_ngrams() here
def count_ngrams(s,n=1):
    """
    find unique substrings(w/ length of n) and its occurances in the given string.
    
    Parameters
    ----------
    s: string provided by the user, exploited within forloop to update dictionary. 
    n: int, determines the length of the substring. Default to 1.
    
    Return 
    ----------
    D: Dictionary containing unique substrings (with length of n) from string s, 
        and its occurances as value.
    """
    D={} 
    for i in range(len(s)-(n-1)):
    # subtract (n-1) from string length to prevent out of range error.
        substr=s[i:i+n]
        
        if substr in D:
        # take the count of substring if it alreay exists in dictionary
            count = D[substr]
        else:
        # if substring is not in dictionary, set a new count
            count = 0
            
        # update the dictionary with incremented count
        D.update({substr:count+1})
        
    return D

count_ngrams("tortoise", n = 2)
```




    {'to': 2, 'or': 1, 'rt': 1, 'oi': 1, 'is': 1, 'se': 1}



## Exercise 3

Now we are going to use our `n`-grams to generate some fake text according to a Markov model. Here's how the Markov model of order `n` works: 

### A. Compute (`n`+1)-gram occurrence frequencies

We have already done this in Exercise 2!  

### B. Pick a starting (`n`+1)-gram

The starting (`n`+1)-gram can be selected at random, or the user can specify it. 

### C. Generate Text

Now we generate text one character at a time. To do so:

1. Look at the most recent `n` characters in our generated text. Say that `n = 3` and the 3 most recent character are `the`. 
2. We then look at our list of `n+1`-grams, and focus on grams whose first `n` characters match. Examples matching `the` include `them`, `the `, `thei`, and so on. 
3. We pick a random one of these `n+1`-grams, weighted according to its number of occurrences. 
4. The final character of this new `n+1` gram is our next letter. 

For example, if there are 3 occurrences of `them`, 4 occurrences of `the `, and 1 occurrences of `thei` in the n-gram dictionary, then our next character is `m` with probabiliy 3/8, `[space]` with probability 1/2, and `i` with probability `1/8`. 

**Remember**: the ***3rd***-order model requires you to compute ***4***-grams. 

## What you should do

Write a function that generates synthetic text according to an `n`-th order Markov model. It should have the following arguments: 

- `s`, the input string of real text. 
- `n`, the order of the model. 
- `length`, the size of the text to generate. Use a default value of 100. 
-  `seed`, the initial string that gets the Markov model started. I used `"Emma Woodhouse"` (the full name of the protagonist of the novel) as my `seed`, but any subset of `s` of length `n+1` or larger will work. 

Demonstrate the output of your function for a couple different choices of the order `n`. 


## Expected Output

Here are a few examples of the output of this function. Because of randomness, your results won't look exactly like this, but they should be qualitatively similar. 

```python
markov_text(s, n = 2, length = 200, seed = "Emma Woodhouse")
```
```
Emma Woodhouse ne goo thimser. John mile sawas amintrought will on I kink you kno but every sh inat he fing as sat buty aft from the it. She cousency ined, yount; ate nambery quirld diall yethery, yould hat earatte
```
```python
markov_text(s, n = 4, length = 200, seed = "Emma Woodhouse")
```

```
Emma Woodhouse!”—Emma, as love,            Kitty, only this person no infering ever, while, and tried very were no do be very friendly and into aid,    Man's me to loudness of Harriet's. Harriet belonger opinion an
```

```python
markov_text(s, n = 10, length = 200, seed = "Emma Woodhouse")
```

```
Emma Woodhouse's party could be acceptable to them, that if she ever were disposed to think of nothing but good. It will be an excellent charade remains, fit for any acquainted with the child was given up to them.
```

## Notes and Hints

***Hint***: A good function for performing the random choice is the `choices()` function in the `random` module. You can use it like this: 

```python
import random

options = ["One", "Two", "Three"]
weights = [1, 2, 3] # "Two" is twice as likely as "One", "Three" three times as likely. 

random.choices(options, weights) 
```

```
['One'] # output
```

The first and second arguments must be lists of equal length. Note also that the return value is a list -- if you want the value *in* the list, you need to get it out via indexing.  

***Hint***: The first thing your function should do is call `count_ngrams` above to generate the required dictionary. Then, handle the logic described above in the main loop.

## Solution


```python

# import time library to calculate the runtime 
import random
import time

# write markov_text() here
def markov_text(s, n = 1, length = 100, seed = None):
    """
    Randomly generate text according to n-th order Markov model. 
    
    Parameters
    ----------
    s: string provided by the user. Used to create n+1 gram dictionary. 
    n: int, determines the length of the substring. Default to 1.
    legnth: legnth of returned generated_text. Default to 100.
    seed: the initial string that gets the Markov model started. Default to None. 
    
    Return 
    ----------
    generated_text: Randomly generated text by random.choices(option, weight).
    """
    start = time.time()
    n_plus_grams = count_ngrams(s, n+1)
    generated_text=""
    
    if seed is None:
    # if user haven't specified the seed, randomly choose the starting n+1gram. 
        option=list(n_plus_grams.keys())
        weight=list(n_plus_grams.values())
        generated_text+=random.choices(option,weight)[0]
    generated_text+=seed

    while len(generated_text)<=length:
    # whileloop writes charcter one by one until it meets the provided length
        options=[]
        weights =[]
        recent_n = generated_text[-n:] 
        # recnent_n is the last n-characters from the generated string
        
        for i in n_plus_grams:
        # loops over n+1 gram dictionary keys and checks if first n-characters
        # matches the recent n-chracters from generated string. 
            if recent_n in i[:n]:
            # if matches, append to random choice option & weight lists
                options.append(i)
                weights.append(n_plus_grams[i])
        generated_text += random.choices(options, weights)[0][-1]
        # add the very last chracter of the chosen n+1 substring. 
        
    end = time.time()
    print(f"-- text generation took: {round(end-start,3)} seconds when n={n} --")
    # print out the runtime 
    
    return   generated_text

markov_text(s, n = 2, length = 200, seed = "Emma Woodhouse")
```

    -- text generation took: 0.146 seconds when n=2 --





    'Emma Woodhouse fat hat ye!—antrow thavextrave he conly Isam mat has marmen al shey, you pand fery bousery dially of you do not nat quardal not wo some makintlectle it inut her, hurend beend his reacer.'




```python
# try out your function for a few different values of n
print(markov_text(s, n = 1, length = 200, seed = "Emma Woodhouse"))
print(markov_text(s, n = 3, length = 200, seed = "Emma Woodhouse"))
print(markov_text(s, n = 4, length = 200, seed = "Emma Woodhouse"))
```

    -- text generation took: 0.088 seconds when n=1 --
    Emma Woodhouseisle he somive the hmad k nt sod, bu, ke wods Dis and a arore vet at Shed beysmo cenathilandil t t ne sathenx adit, ivan dange eapas hedllt hal, t pe.” ad se be g is ve atra wenf alalinis
    -- text generation took: 0.302 seconds when n=3 --
    Emma Woodhousember such grave your leas, fount ford at him glady exactobery fort of all ever one yout companion she mance; and who lives a revery soon the reign old—a ver in to Lond real ratisfield, sh
    -- text generation took: 0.595 seconds when n=4 --
    Emma Woodhouse. She know nicer that he had some,” she hourly how should not at all man.”“Well, and sort one is they had near there is judge (that revening to me any body who have brother. The afforded 



```python
# try another value of n! 
print(markov_text(s, n = 7, length = 200, seed = "Emma Woodhouse"))
```

    -- text generation took: 1.62 seconds when n=7 --
    Emma Woodhouse,” was her youth is determination range and would never had. This happier flow of his being in the yesterday.”“I do think him so much above my charades. Courtship, you know. Proportionles



```python
# try third value! 
print(markov_text(s, n = 10, length = 200, seed = "Emma Woodhouse"))
```

    -- text generation took: 2.175 seconds when n=10 --
    Emma Woodhouse, he would not put off for any inducement to her husband could not remember nothing;—not even that particular a meaning in this compliment,” said Mr. Knightley, laughing again. “I do not 


## Exercise 4

Using a `for`-loop, print the output of the function for `n` ranging from `1` to `10`. 



```python
# code here
for n in range(1,11):
    print(markov_text(s, n, length = 200, seed = "Emma Woodhouse"))
    print()
```

    -- text generation took: 0.085 seconds when n=1 --
    Emma Woodhouse he, shoofunod be iss. tithil!” yot d t,”“Thitharsthed, nshoen s d'stofre he o houlo ofo asher a—sstche want s abuedoty btond veelle m, sut shany bupouppes I wnd it obeatote bedo ofomadis
    
    -- text generation took: 0.13 seconds when n=2 --
    Emma Woodhouse forecess the sawitent ing forthre Mr. Batiout. Bat thor you ded,”—unposs ce ch of her con Emmes the coughtlearromadeen way did iss nableal and houbjections, and a vis of? A put per hores
    
    -- text generation took: 0.32 seconds when n=3 --
    Emma Woodhouse; “and prehen home,” this she people forwards the could I am gladied. By ter her being of here acquirece ove man over far thould beyond to a genting to been I knew his she; “a made up—One
    
    -- text generation took: 0.594 seconds when n=4 --
    Emma Woodhouse's good woman in comparing a sense, that sort of but have it would their had nevery must be the fetch other tea case him; but her for sure soften for to feel it, by that's pays dispare sh
    
    -- text generation took: 0.936 seconds when n=5 --
    Emma Woodhouse, is attachment to have a little gallant confused; said Mr. Elton—“I do not be so much an always mention, and always like the private enjoyment by no more made a particularly then doors; 
    
    -- text generation took: 1.29 seconds when n=6 --
    Emma Woodhouse has every next eight-and-thirty, was they were to delay in one of the time, as she means his own causes, from they would not have time,” said Mr. Woodhouse of their opinions while she wa
    
    -- text generation took: 1.584 seconds when n=7 --
    Emma Woodhouse could.A complete picture?”Harriet would ask for the gratitude and imitations too high. Miss Taylor for you abusing them, he had then afterwards made her grandpapa, can you give me such a
    
    -- text generation took: 1.824 seconds when n=8 --
    Emma Woodhouse, do you think so?” replied, “that she had spent two very gentle of late hour to call upon us. I am quite determined to one another, to her husbands and wives in the evening must be doing
    
    -- text generation took: 2.016 seconds when n=9 --
    Emma Woodhouse's feelings were displayed. Miniatures, half-lengths, pencil, crayon, and was all eat up. His ostensible reason, however, and rich, with a smiling graciously, “would I advise you either w
    
    -- text generation took: 2.152 seconds when n=10 --
    Emma Woodhouse, what a pity that they would eat.Such another small basin of thin gruel as his own; for as to Frank, it was more equal to the match, as you call it, means only your planning it. She fear
    


### Few Observations
1. How does the generated text depend on `n`? 
2. How does the time required to generate the text depend on `n`? 

As the n increase, the generated text becomes more readable. This is due to the longer length substring of n+1 gram dictionary keys. By enabling longer substrings, we have higher chance of completing the word or sentence. 
For instance of generating 'are', when n=1 and most recent n is 'a', n+1 gram options may be, 'a.', 'ar', or 'a '. There is a higher chance of altering the intended word or syntax after letter 'a'. However, when n = 6, the most recent n might be 'They a'. Then n+1 gram options might be 'They as(k)', 'They ar(e)', 'They al(l)' and so on. We would still have hard time meeting the exact target word that would fit the context, still, it would more likely to complete word and make the texts more readable. 

The time required for generation increases with n because there are more keys in n+1 gram dictionaries as the occurrences of each keys would decrease. 'ar' will occur much often than 'They are re' within the provided string. To specify with our example string, when n=2, n+1 grams have 5490 keys. However, when n=10, n+1 grams have 137892 keys, that is 132402 more pairs within dictionary. As the dictionary gets longer, loops accessing dictionary data will require longer runtime. This is why we see the longer time required to generate the text. 
