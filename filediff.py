import difflib

directory1='/Users/shrinix/git/medium/Document Similarities/'
directory2='/Users/shrinix/git/medium/Document Similarities/'

file1 = 'orig.drl' #'Document1.txt'
file2 = 'gen.drl' #'Document2.txt'
lines1 = []
lines2 = []

with open(directory1+file1) as f:
    linesx = f.read().splitlines()
    for line in linesx:
        lines1.append(line)

with open(directory2+file2) as f:
    #lines2 = [line.rstrip() for line in f]
    linesx = f.read().splitlines()
    for line in linesx:
        lines2.append(line)

diff = difflib.unified_diff(lines1, lines2, fromfile='file1', tofile='file2', lineterm='', n=0)
lines = list(diff)[2:]
added = [line[1:] for line in lines if line[0] == '+']
removed = [line[1:] for line in lines if line[0] == '-']

print('additions: fromFile='+file1+' -> toFile='+file2)
for line in added:
    print(line)
print('-------------------')
print('removals: fromFile='+file1+' -> toFile='+file2)
for line in removed:
    print(line)
# print('additions, ignoring position')
# for line in added:
#     if line not in removed:
#         print(line)