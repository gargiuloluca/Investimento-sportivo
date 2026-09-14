# Sports Analytics & Value Scanner (GitHub App)

Applicazione autonoma per l'analisi e il confronto delle probabilità sportive, progettata per operare interamente a costo zero su GitHub Actions e GitHub Pages.

## Mercati Analizzati per Ogni Partita
1. **Esito Finale:** 1, X, 2
2. **Doppia Chance:** 1X, X2, 12
3. **Gol:** Under 2.5 e Over 2.5 (Modello Poisson Bivariato)
4. **Calci d'Angolo:** Over 8.5 / Under 11.5
5. **Cartellini (Gialli/Rossi):** Over 3.5 / Under 5.5
6. **Migliore Singola Evidenziata:** Individua automaticamente l'esito con la probabilità percentuale più alta in assoluto.

## Istruzioni di Installazione
1. Crea un nuovo repository su GitHub (es. `investimento-sportivo`).
2. Carica tutti i file mantenendo questa struttura di cartelle.
3. Abilita **GitHub Actions** con permessi di scrittura:
   - *Settings* -> *Actions* -> *General* -> *Workflow permissions* -> Seleziona **"Read and write permissions"** e clicca *Save*.
4. Abilita **GitHub Pages** per consultare la Dashboard:
   - *Settings* -> *Pages* -> Sotto *Build and deployment* seleziona Branch: `main` e cartella: `/(root)`.
5. Vai nella scheda **Actions**, seleziona *Daily Sports Analytics Scanner* e premi **Run workflow** per avviare subito la prima analisi!
