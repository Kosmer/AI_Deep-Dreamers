import os

input_folder = "path/della/cartella"
output_file = "result.txt"

with open(output_file, 'w', encoding="utf8") as output:
            for filename in os.listdir(input_folder):
                if filename.endswith(".txt"):
                    with open(os.path.join(input_folder, filename), 'r', encoding="utf8") as file:
                        content = file.read()
                        output.write(content + "\n*\n")
        print(f"I file nella cartella sono stati uniti con successo in {output_file}")