# -*- coding: utf-8 -*-
#file for removing unnecessary punctuations and limiting the length of abstract
#after 60 characters, the abstract is limited to the nearest word.
import regex as re
with open("26testing_1.csv","r") as f:
    with open("27testing.csv","w") as w:
        for line in f:
            p1,p2=line.split("\t")
            p2=re.sub(r',|;|\"|\\','',p2)
            p2=re.sub(r'( $)','',p2)
            p2=re.sub(r'\([^)]+\)|\[[^\[]+\]','',p2,re.UNICODE)
            p2=p2.replace(')','')
            p2=re.sub(r'  |   ',' ',p2)
            p2=re.sub(r'.$','',p2)

            if len(p2)>=70:
                p2=re.sub(r"(?<=.{60}\b)\s(.+)",'',p2)
                w.write(p1+"\t"+p2)
            else:
                w.write(p1+"\t"+p2)