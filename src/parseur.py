import os.path


def parser_entrainement():
    """Lit le fichier data/train.csv
    et renvoit les données d'entraînement sous la forme
    d'une liste sous la forme [cible, var1, var2, ...]"""
    fich = open(os.path.join("data", "train.csv"))
    next(fich)
    donnees = []
    for ligne in fich:
        temp = ligne.replace("\n", "").split(",")
        tmp = []
        tmp.append(temp[1])
        for x in temp[2:]:
            tmp.append(float(x))
        donnees.append(tmp)
    return donnees


def parser_test():
    """Lit le fichier data/test.csv
    et renvoit les données de test sous la forme
    d'une liste sous la forme [id, var1, var2, ...]"""
    fich = open(os.path.join("data", "test.csv"))
    next(fich)
    donnees = []
    for ligne in fich:
        temp = ligne.replace("\n", "").split(",")
        tmp = []
        tmp.append(temp[0])
        for x in temp[1:]:
            tmp.append(float(x))
        donnees.append(tmp)
    return donnees
