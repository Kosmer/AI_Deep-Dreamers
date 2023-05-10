with open('Saved_Chat_GPT_poesie.txt', 'r') as f:
    with open('new_file.txt', 'w') as new_file:
       
        lines = f.readlines()
        for i in range(0, len(lines)):
            if lines[i].isspace():
                if lines[i+1][0] == "*":
                    continue
                elif lines[i-1][0] == "*":
                    continue
                else:
                    new_file.write(lines[i])
            else:
                new_file.write(lines[i])


with open('new_file.txt', 'r') as f:
    with open('poesiegpt.txt', 'w') as new_file2:
        
        lines = f.readlines()
        for i in range(0, len(lines)):
            
            if lines[i][0] == "*":
                new_file2.write("*\n")
            else:
                new_file2.write(lines[i])
            

with open('poesiegpt.txt', 'r',  encoding='utf-8') as inp:
    with open('autori_gpt.txt', 'w') as out:
        
        lines = inp.readlines()
        for i in range(0, len(lines)):
            
            if lines[i][0] == "*":
                out.write("1\n")


with open('aldamerini.txt', 'r',  encoding='utf-8') as inp:
    with open('autori_aldamerini.txt', 'w') as out:
        
        lines = inp.readlines()
        for i in range(0, len(lines)):
            
            if lines[i][0] == "*":
                out.write("0\n")
        
        
                