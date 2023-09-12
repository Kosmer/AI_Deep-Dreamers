# Nome dei file di input
file1_nome = "file1.txt"                    #file1 = file con le poesie originali
file2_nome = "file2.txt"                    #file2 = file con le posie chatgpt

# Nome del file di output
file_output_nome = "file_unito.txt"                                     #txt poesie unite
file_cont_poesie_originali = "numero_poesie_originali.txt"              #txt numero poesie di autori originali
file_cont_poesie_chatgpt = "numero_poesie_ChatGPT.txt"                  #txt numero poesie di chatgpt
file_output_cont_poesie = "autori.txt"


# Leggi il contenuto del primo file
with open(file1_nome, 'r') as file1:
    with open(file_cont_poesie_originali, "r+") as poesie_originali:
        for linea in file1:
            if linea[0]=="*":
                poesie_originali.write("0\n")
        contenuto_file1 = file1.read()
        contenuto_poesie_originali = poesie_originali.read()

# Leggi il contenuto del secondo file
with open(file2_nome, 'r') as file2:
    with open(file_cont_poesie_chatgpt, "r+") as poesie_chatgpt:
        for linea in file2:
            if linea[0]=="*":
                poesie_chatgpt.write("1\n")
        contenuto_file2 = file2.read()
        contenuto_poesie_chatgpt = poesie_chatgpt.read()

# Unisci i contenuti dei due file
contenuto_unito = contenuto_file1 + contenuto_file2
contenuto_autori = contenuto_poesie_originali + contenuto_poesie_chatgpt



# Scrivi il contenuto unito nel file di output
with open(file_output_nome, 'w') as file_output:
    file_output.write(contenuto_unito)

with open(file_output_cont_poesie, 'w') as file_output:
    file_output.write(contenuto_autori)

print(f"Il contenuto dei file {file1_nome} e {file2_nome} è stato unito nel file {file_output_nome}.")
