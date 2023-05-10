import re




if __name__ == '__main__':
    with open("./source.txt", 'r', encoding="utf8") as file:
        text = file.read()
        text = text.replace("<p>", "")
        ntext = text.split('\n')
        ris = []
        substring = "<span style=\"font-family: \'book antiqua\', palatino, serif;\">"                  #STRINGA DA TROVARE CHE CONTIENE IL TESTO
        i = -1
        men = 0
        substring5 = "</span><br />"
        substring6 = "</span></p>"
        
        cont = 0
        while i<len(ntext)-1:
            i+=1
            if substring in ntext[i]:
                contrighe = 1
                txt = ""
                match=(re.search(substring, ntext[i]))                                              #CERCA SUBSTRING NELLA RIGA NTEXT
                men = -13
                s = ntext[i][match.end():men]

                
                i+=1
                while substring in ntext[i]:
                    contrighe+=1
                    if s[0]==" ":
                        s = s.replace(" ", "", 1)
                    txt+=s
                    txt+="\n"
                    match=(re.search(substring, ntext[i]))
                    match2=(re.search(substring5, ntext[i]))
                    if match2!=None:
                        men = -13
                    else:
                        men = -11

                    s = ntext[i][match.end():men]
                    i+=1

                txt = txt.replace("&#8217;", "'")
                txt+="*\n"
                if contrighe>3:
                    ris.append(txt)

                

        with open("resultscan.txt", "w") as f:
            for i in range(0, len(ris)):
                #print(ris[i])
                f.write(ris[i])

        with open("resultscan.txt", "r") as f2:
            text = file.read()
            ntext2 = text.split('\n')
            contstar = 0
            for i in range(0, len(ntext2)-1):
                if ntext[i]=='*':
                    contstar+=1
                else:
                    contstar=0

                if contstar>1:
                    ris.pop(i)

        with open("resultscan.txt", "w") as f:
            for i in range(0, len(ris)):
                f.write(ris[i])


        print(len(ris))
        