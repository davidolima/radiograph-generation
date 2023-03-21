"""
David Lima - 22/11/22
"""

import os, sys
import pandas as pd

def gerarDF(diretorio, outname):
    images = os.listdir(diretorio)

    dados = {"file":[], "number":[], "pid":[], "fileid": [],
           "date": [], "sex": [], "age":[]}

    for file in images:
       dados["file"].append(file)
       file = file.split('-')
       dados["number"].append(file[1])
       dados["pid"].append(file[3])
       dados["fileid"].append(file[5])
       dados["date"].append(file[7])
       dados["sex"].append(file[10])
       dados["age"].append(file[11])

    return pd.DataFrame(data=dados)

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        images_dir = "/media/david/bb5b899e-a7e9-49dd-b267-5014035bf700/datasets/pan-radiographs/1st-set/images"
        nome_out = "pan-radiographs.csv"
    else:
        images_dir = sys.argv[1]
        nome_out = sys.argv[2]

    print(f"\nDiretório: {images_dir}")
    print(f"Arquivo de saída: {nome_out}\n")

    df = gerarDF(images_dir, nome_out)
    print(df)
    df.to_csv(nome_out, index=False)

    print("Pronto. Pressione qualquer tecla para sair...")
    input()