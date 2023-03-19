directory1='./'
directory2='./'

file1 = 'orig.drl' #'Document1.txt'
file2 = 'gen.drl' #'Document2.txt'

def load_files():
    lines1 = []
    lines2 = []
    indexes1 = []
    indexes2 = []
    lines = []
    indexes = []
    short_col_name1 = []
    short_col_name2 = []
    short_col_names = []

    # removing the new line characters
    with open(directory1+file1) as f:
        i = 0
        lines = f.read().splitlines()
        for line in lines:
            lines1.append(line)
            indexes1.append(file1 + '('+str(i)+')->'+line)
            short_col_name1.append(file1+'.'+str(i))
            i=i+1

    with open(directory2+file2) as f:
        #lines2 = [line.rstrip() for line in f]
        i = 0
        lines = f.read().splitlines()
        for line in lines:
            lines2.append(line)
            indexes2.append(file2 + '('+str(i)+')->'+line)
            short_col_name2.append(file2+'.'+str(i))
            i=i+1

    lines = lines1+lines2
    indexes = indexes1+indexes2
    short_col_names = short_col_name1+short_col_name2

    return lines, indexes, short_col_names