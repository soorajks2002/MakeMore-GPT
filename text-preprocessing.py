names = open('names.txt', 'r').read().splitlines()

characters = sorted(list(set(''.join(names))))
ctoi = {c: i for i, c in enumerate(characters)}
ctoi['<S>'] = 26
ctoi['<E>'] = 27

text_data = []

for name in names:
    name = ['<S>'] + list(name) + ['<E>']
    text_data.append(name)

print(text_data[0])
