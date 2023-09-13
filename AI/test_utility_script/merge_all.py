import os

# Cartella contenente i file TXT
cartella_input = "./dataset"

# Nome dei file di output
file_output1 = "poesie_ChatGPT.txt"
file_output2 = "poesie_originali.txt"

# Inizializza una lista per contenere il contenuto dei file da unire
contenuto_da_unire1 = []
contenuto_da_unire2 = []

# Elabora i file nella cartella di input
for nome_file in os.listdir(cartella_input):
    if nome_file.startswith("ChatGPT") and nome_file.endswith(".txt"):
        # Verifica che il file inizi con "chatgpt" e termini con ".txt"
        percorso_completo_file = os.path.join(cartella_input, nome_file)
        with open(percorso_completo_file, "r", encoding="utf-8-sig") as file_da_leggere:
            contenuto_da_unire1.append(file_da_leggere.read())


for nome_file in os.listdir(cartella_input):
    if nome_file.startswith("Originale") and nome_file.endswith(".txt"):
        # Verifica che il file inizi con "Originale" e termini con ".txt"
        percorso_completo_file = os.path.join(cartella_input, nome_file)
        with open(percorso_completo_file, "r", encoding="utf-8-sig") as file_da_leggere:
            contenuto_da_unire2.append(file_da_leggere.read())

# Unisci il contenuto dei file in un unico testo
contenuto_unificato1 = "\n".join(contenuto_da_unire1)
contenuto_unificato2 = "\n".join(contenuto_da_unire2)

# Scrivi il contenuto unificato nel file di output
percorso_completo_output1 = os.path.join(cartella_input, file_output1)
with open(percorso_completo_output1, "w", encoding="utf-8-sig") as file_output1:
    file_output1.write(contenuto_unificato1)


percorso_completo_output2 = os.path.join(cartella_input, file_output2)
with open(percorso_completo_output2, "w", encoding="utf-8-sig") as file_output2:
    file_output2.write(contenuto_unificato2)

print("Unione autori effettuata")


# Nome dei due file di input da fare il merge
file_poesie_originali = "dataset/poesie_originali.txt"                     #file con le poesie originali
file_poesie_chatgpt = "dataset/poesie_ChatGPT.txt"                          #file con le posie chatgpt

# Nome dei file di output. 
file_poesie_merge = "poesie.txt"                                        #txt con tutte le poesie unite
file_label_poesie_originali = "numero_poesie_originali.txt"              #txt contenente un numero di zeri (label) in base a quante poesie abbiamo degli autori originali 
file_label_poesie_chatgpt = "numero_poesie_ChatGPT.txt"                  #txt contenente un numero di zeri (label) in base a quante poesie abbiamo di chatgpt
file_label_merge = "autori.txt"                                         #txt con le label


with open(file_poesie_originali, 'r', encoding="utf-8-sig") as file1:
    contenuto_poesie_generali = file1.read()
    if contenuto_poesie_generali[-1] != '\n':
        contenuto_poesie_generali += '\n'


# Memorizza il contenuto del file delle poesie degli esseri umani. Successivamente scorre il file, quando trova un * scrive uno 0 (label per esseri umani). 
with open(file_poesie_originali, 'r', encoding="utf-8-sig") as file1:
    with open(file_label_poesie_originali, 'w', encoding="utf-8-sig") as poesie_originali:
        with open(file_label_poesie_originali, "r+", encoding="utf-8-sig") as poesie_originali:
            for linea in file1:
                if linea[0]=="*":
                    poesie_originali.write("0\n")
            

# Memorizza le label scritte nel file
with open(file_label_poesie_originali, "r+", encoding="utf-8-sig") as poesie_originali:
            label_poesie_originali = poesie_originali.read()


with open(file_poesie_chatgpt, 'r', encoding="utf-8-sig") as file2:
    contenuto_poesie_chatgpt = file2.read()

# Memorizza il contenuto del file delle poesie di chatgpt. Successivamente scorre il file, quando trova un * scrive un 1 (label per chatgpt). 
with open(file_poesie_chatgpt, 'r', encoding="utf-8-sig") as file2:
    with open(file_label_poesie_chatgpt, 'w', encoding="utf-8-sig") as poesie_chatgpt:
        with open(file_label_poesie_chatgpt, "r+", encoding="utf-8-sig") as poesie_chatgpt:
            for linea in file2:
                if linea[0]=="*":
                    poesie_chatgpt.write("1\n")
            

# Memorizza le label scritte nel file
with open(file_label_poesie_chatgpt, "r+", encoding="utf-8-sig") as poesie_chatgpt:
            label_poesie_chatgpt = poesie_chatgpt.read()

# Unisce i contenuti dei file delle poesie e dei file delle label
poesie_unite = contenuto_poesie_generali + contenuto_poesie_chatgpt
label_unite = label_poesie_originali + label_poesie_chatgpt

# Scrivi il contenuto unito nel file di output
with open(file_poesie_merge, 'w', encoding="utf-8-sig") as file_output:
    file_output.write(poesie_unite)

with open(file_label_merge, 'w', encoding="utf-8-sig") as file_output:
    file_output.write(label_unite)


# Rimuove i file parziali con le label parziali (solo chatgpt e solo esseri umani)
os.remove(file_label_poesie_originali)
os.remove(file_label_poesie_chatgpt)

print(f"Il contenuto dei file di poesie è stato unito nel file {file_poesie_merge}. Il file delle label è stato generato col nome {file_label_merge}")

