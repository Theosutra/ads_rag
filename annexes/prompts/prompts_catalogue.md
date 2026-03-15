# Catalogue des prompts — Mémoire RAG ADS CESI

## Prompt Standard (concaténation brute)

```
Utilise les passages suivants pour répondre à la question.
Si la réponse ne figure pas dans les passages, réponds 'Je ne sais pas'.

Passages :
{passage_1}
{passage_2}
...
{passage_n}

Question : {question}
Réponse :
```

**Usage** : Configuration RAG_standard, protocoles A, B, C.

---

## Prompt Citations strictes

```
Réponds à la question en te basant UNIQUEMENT sur les passages numérotés ci-dessous.
Pour chaque affirmation, indique le numéro du passage source entre crochets [N].
Si aucun passage ne permet de répondre, réponds exactement 'Je ne sais pas.'

[1] {passage_1}
[2] {passage_2}
...
[k] {passage_k}

Question : {question}
Réponse (avec citations) :
```

**Usage** : Configuration RAG_citations, RAG_verify. Teste H2 et H3.

---

## Prompt Baseline (sans retrieval)

```
Question : {question}
Réponse :
```

**Usage** : Configuration baseline (sans RAG), Protocole A.

---

## Notes d'implémentation

- Le modèle génératif est mT5-large dans **tous** les cas.
- Le prompt est tronqué à 1024 tokens si nécessaire.
- Le marqueur d'abstention est : `Je ne sais pas`
- Pour FQuAD 2.0 (questions impossibles), le taux d'abstention correcte est mesuré.
