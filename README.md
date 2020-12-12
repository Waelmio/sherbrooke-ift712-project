<center>
<h1>Projet de classification de feuilles selon leurs espèces</h1>
<h2>Université de Sherbrooke - IFT712: Techniques d'apprentissage</h2>
</center>
</br>
</br>
<pre>- Bougeard Yann - 20 137 996
- Wilmo Maël    - 20 138 003
</pre>

## Exécution

Pour lancer la classification, utilisez la commande:

```
python3 src/classifier.py model
```

Les modèles disponibles sont:

- ```svm```  pour une classification par **Machine à vecteurs de support**.
- ```mlp``` pour une classification par **Perceptron Multicouche**.
- ```lda``` pour une classification par **Analyse Discriminante Linéaire**.
- ```logistic``` pour une classification par **Régression Logistique**.
- ```ridge``` pour une classification par **Régression de Ridge**.
- ```perceptron``` pour une classification par **Perceptron**.