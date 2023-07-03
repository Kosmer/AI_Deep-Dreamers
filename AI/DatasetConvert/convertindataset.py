with open('poesiegpt.txt', 'r',  encoding="utf8") as file:
    data = file.read()

poems = data.split("*\n")

    
for i, poem in enumerate(poems):
    filename = f"poem_{i+1}.txt"  # Genera un nome di file incrementale
    with open(filename, 'w') as poem_file:
        poem_file.write(poem)