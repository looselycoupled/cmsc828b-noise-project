# Overview

This project extends Khayrallah et al's original work on the impact of various types of noise.  Our exploration attempts to train a transformer under the same conditions as the reference paper to see how a different model is impacted.

> Khayrallah, Huda, and Philipp Koehn. “On the Impact of Various Types of Noise on Neural Machine Translation.” Proceedings of the 2nd Workshop on Neural Machine Translation and Generation, Association for Computational Linguistics, 2018, pp. 74–83. ACLWeb, doi:10.18653/v1/W18-2709.


## Todo

* load saved checkpoint and use it
* modify logging to save to file in addition to console

## Notes

Various snippets for reference

```
# Head 500k lines, replace entities, and then save file
head -n 500000  ../original/baseline.tok.en | perl -MHTML::Entities -pe 'decode_entities($_);' > ./baseline.500000.tok.en


# clean validation set
cat newstest2016-deen-ref.en.sgm | sed 's/<seg id="[0-9]*">//g; s/<\/seg>//g; s/<\/*p>//g; s/<refset.*>//g; s/<\/refset>//g; s/<doc .*>//g; s/<\/doc>//g'  | sed '/^$/d' >newstest2016.en
cat newstest2016-deen-src.de.sgm | sed 's/<seg id="[0-9]*">//g; s/<\/seg>//g; s/<\/*p>//g; s/<srcset.*>//g; s/<\/srcset>//g; s/<doc .*>//g; s/<\/doc>//g'  | sed '/^$/d' >newstest2016.de

# clean test set
cat newstest2017-deen-ref.en.sgm | sed 's/<seg id="[0-9]*">//g; s/<\/seg>//g; s/<\/*p>//g; s/<refset.*>//g; s/<\/refset>//g; s/<doc .*>//g; s/<\/doc>//g'  | sed '/^$/d' >newstest2017.en
cat newstest2017-deen-src.de.sgm | sed 's/<seg id="[0-9]*">//g; s/<\/seg>//g; s/<\/*p>//g; s/<srcset.*>//g; s/<\/srcset>//g; s/<doc .*>//g; s/<\/doc>//g'  | sed '/^$/d' >newstest2017.de


```