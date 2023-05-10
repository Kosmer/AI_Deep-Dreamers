import re
x = "eh *A* ciao *B* vabbe"

trails = ("*A*", "*B*")

# \b means word boundaries.
regex = r"\b(?:{})\b".format("|".join(trails))
print(re.split(regex, x))
