import regex as re
with open("16_allemb.emb","r") as f:
    with open("16_allemb.emb","w") as w:
        p = re.compile('([a-z]|[.]|\))\s(-[0-9][.])|([a-z]|[.]|\))\s([0-9][.])')
        for line in f:
            line=re.sub(r'([a-z]|[.]|\))\s(-*[0-9][.])',r'\1\t\2',line)#([a-z]d{4}\-\d{2})
            w.write(line)
