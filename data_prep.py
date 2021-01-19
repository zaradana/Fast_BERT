# output_filename = './opt/ml/input/data/val.txt'

# fo = open(output_filename, "w")

# fname = './opt/ml/input/data/val_orig.txt'
# f = open(fname, "r")
# for line in f:
#     if not line.startswith('-DOCSTART'):
#         splits = line.split(' ')
#         if splits[0] != '\n':
#             fo.write(splits[0]+" "+splits[-1][:-1]+"\n")
#         else:
#             fo.write(splits[0])
# f.close()

# fo.close()

#%%
# import pandas as pd
# output_filename = './opt/ml/input/data/test.csv'

# fo = open(output_filename, "w")

# fname = './opt/ml/input/data/test.txt'
# text = []
# sent =[]
# f = open(fname, "r")
# for line in f:
#     if not line.startswith('-DOCSTART'):
#         splits = line.split(' ')
#         if splits[0] != "\n":
#             sent.append(splits[0])
#         elif len(sent)>0:
#             text.append(' '.join(sent))
#             sent=[]
# f.close()
# df = pd.DataFrame(text, columns=["text"])
# df.to_csv(output_filename, index=False)
#%%
file_ext = "train.txt"
output_filename = './opt/ml/input/data/'+file_ext

fo = open(output_filename, "w")

fname = './opt_ner/ml/input/data/'+file_ext
text = []
sent =[]
f = open(fname, "r")
for line in f:
    splits = line.split(' ')
    if splits[0] != "\n":
        sent.append(splits[0])
    elif len(sent)>0:
        fo.write(' '.join(sent)+"\n")
        sent=[]
f.close()
# df = pd.DataFrame(text, columns=["text"])
# df.to_csv(output_filename, index=False)

fo.close()

# %%
