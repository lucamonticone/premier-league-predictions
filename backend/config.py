

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional



BACKEND_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = BACKEND_DIR.parent



DATA_DIR: Path = PROJECT_ROOT / "data"


DEFAULT_COMPETITION_CODE: str = "PL"


DEFAULT_SEASON_YEAR: int = 2025

DEFAULT_SEASON_LABEL: str = "2025-2026"


def _season_label_to_suffix(season_label: str) -> str:
    """
    Converte '2025-2026' ->  '2025_2026' per usarlo comodamente nei nomi file.
    """
    return season_label.replace("-", "_")


PREMIER_RESULTS_CSV: Path = (
    DATA_DIR / f"premier_results_raw_{_season_label_to_suffix(DEFAULT_SEASON_LABEL)}.csv"
)


TEAMS_FILE: Path = (
    DATA_DIR / f"teams_premier_{_season_label_to_suffix(DEFAULT_SEASON_LABEL)}.xlsx"
)


PREMIER_OUTCOMES_DIR: Path = DATA_DIR


MATCHDAY_PREDICTIONS_JSON: Path = DATA_DIR / "matchday_predictions.json"



FOOTBALL_DATA_TOKEN_ENV_VAR:  str = "FOOTBALL_DATA_API_TOKEN"


ENV_FILE: Path = PROJECT_ROOT / "footballAPI.env"


def _load_env_file(path: Optional[Path] = None) -> None:
    """
    Carica variabili d'ambiente da un file tipo .env se esiste.

    Formato atteso per ogni riga:
        NOME_VARIABILE=valore

    Le righe vuote o che iniziano con '#' sono ignorate.
    Se una variabile è già presente in os.environ, non viene sovrascritta.
    """
    env_path = path or ENV_FILE
    if not env_path.is_file():
        return

    try:
        text = env_path.read_text(encoding="utf-8")
    except OSError:
        return

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


def get_football_data_api_token(env_var: str = FOOTBALL_DATA_TOKEN_ENV_VAR) -> str:
    """
    Restituisce il token API per football-data.org.

    Ordine di ricerca:
    1. Variabile d'ambiente (FOOTBALL_DATA_API_TOKEN di default).
    2. File 'footballAPI.env' nella root del progetto.

    Se non viene trovato, alza un RuntimeError con un messaggio chiaro.
    """
    # 1) Se non è già impostato, provo a caricare dal file .env
    if env_var not in os.environ:
        _load_env_file()

    token = os.environ.get(env_var)
    if not token:
        raise RuntimeError(
            f"Token API non trovato. Imposta la variabile d'ambiente '{env_var}' "
            f"oppure crea un file '{ENV_FILE.name}' nella root del progetto con una riga:\n"
            f"{env_var}=IL_TUO_TOKEN"
        )
    return token


def ensure_data_dir_exists() -> None:
    """
    Crea la cartella DATA_DIR se non esiste già.

    Non fallisce se la directory esiste già (exist_ok=True).
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
