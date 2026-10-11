# SmartDoc

## Lancer l'application avec Docker

Depuis la racine du projet :

```powershell
docker compose up --build
```

L'application est accessible sur <http://localhost:8501>. Ollama reste
uniquement accessible aux services du réseau Docker et l'application est
publiée sur l'interface locale de la machine via un proxy Nginx. Les sessions
sont conservées dans un cookie `HttpOnly`, non dans l'URL.

Si tu places Nginx derrière un accès HTTPS, active aussi l'attribut `Secure`
du cookie :

```powershell
$env:SMARTDOC_COOKIE_SECURE = 'true'
docker compose up --build
```

Ne publie pas directement les ports internes de Streamlit ou du service de
session.

La base `smartdoc.db` et les documents sous `app/data/` sont des données
privées locales : ils ne doivent jamais être ajoutés à Git ni à une image
Docker. Les configurations Git et Docker les excluent. Les montages Compose
conservent ces données sur le disque hôte lorsque le conteneur est supprimé.
Ne lance pas `docker compose down -v` pour supprimer un conteneur : cela
supprime les volumes Docker (notamment les modèles), même si les montages de
données locaux restent présents.

### Index GraphRAG

En plus des embeddings, chaque espace utilisateur conserve un graphe SQLite
dans `app/data/users/<id>/.knowledge_graph.sqlite3`. À l'import, Ollama extrait
les entités et relations explicitement présentes dans le document en arrière-plan,
en un seul traitement par document. Les types courants (comptes rendus, courriels,
factures, procédures, etc.) sont classés par règles rapides avant tout appel au
modèle de nommage. Le chatbot combine ensuite la recherche vectorielle avec les
passages reliés par ces entités, y compris dans des catégories différentes.
Le fichier du graphe est privé comme les documents et doit être inclus dans les
sauvegardes du dossier `app/data/users`.

### Sauvegarder et transférer les données

Pour déplacer un compte vers un autre ordinateur, arrête d'abord l'application,
puis sauvegarde **ensemble** la base SQLite et les documents ; ils contiennent
les comptes et les fichiers correspondants :

```powershell
docker compose down
$backup = Join-Path $HOME 'SmartDoc-backup'
New-Item -ItemType Directory -Force $backup | Out-Null
$dbPath = '.\smartdoc.db'
if (Test-Path $dbPath -PathType Container) {
    $dbPath = Join-Path $dbPath 'smartdoc.db'
}
Copy-Item $dbPath (Join-Path $backup 'smartdoc.db')
Copy-Item .\app\data\users (Join-Path $backup 'users') -Recurse
```

Transfère cette sauvegarde par un canal privé et sûr, puis restaure le fichier
de base et le dossier `users` aux mêmes emplacements sur l'autre ordinateur
**avant** le premier `docker compose up`. Si l'application a déjà été lancée
sur le nouvel ordinateur sans base, Docker peut avoir créé `smartdoc.db` comme
dossier ; dans ce cas, restaure le fichier dans `smartdoc.db\smartdoc.db`.
Ne partage pas la sauvegarde avec le dépôt public.

### Données précédemment publiées dans l'historique Git

Des versions antérieures du dépôt ont pu contenir des fichiers sous
`app/data/`. Les supprimer de la version actuelle ne les retire pas des anciens
commits déjà publiés. Si ces fichiers contenaient des données réelles, considère
qu'elles ont été divulguées et demande une purge coordonnée de l'historique
public ; une telle réécriture invalide les clones et doit être coordonnée avec
tous les contributeurs.