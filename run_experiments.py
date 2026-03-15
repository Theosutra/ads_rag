"""Point d'entrée principal pour l'exécution des expériences RAG."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("reports/logs/experiments.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

SEED = 42


def set_env_seeds() -> None:
    """Configure les seeds d'environnement pour la reproductibilité totale."""
    os.environ["RANDOM_SEED"] = str(SEED)
    os.environ["TORCH_SEED"] = str(SEED)
    os.environ["NUMPY_SEED"] = str(SEED)
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    logger.info("Seeds fixées à %d", SEED)


def run_protocol_a(lang: str, max_samples: int | None) -> None:
    from src.experiments.protocol_a import ProtocolA
    protocol = ProtocolA()
    protocol.run(lang=lang, max_samples=max_samples)


def run_protocol_b(lang: str, max_samples: int | None) -> None:
    from src.experiments.protocol_b import ProtocolB
    protocol = ProtocolB()
    protocol.run(lang=lang, max_samples=max_samples)


def run_protocol_c(lang: str, max_samples: int | None) -> None:
    from src.experiments.protocol_c import ProtocolC
    protocol = ProtocolC()
    protocol.run(lang=lang, max_samples=max_samples)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exécution des expériences RAG — Mémoire ADS CESI 2025-2026",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python run_experiments.py --protocol A --lang fr
  python run_experiments.py --protocol A --lang en
  python run_experiments.py --protocol B --lang fr --max-samples 100
  python run_experiments.py --protocol C
  python run_experiments.py --protocol all
        """,
    )
    parser.add_argument(
        "--protocol",
        choices=["A", "B", "C", "all"],
        required=True,
        help="Protocole expérimental à exécuter",
    )
    parser.add_argument(
        "--lang",
        choices=["fr", "en", "both"],
        default="both",
        help="Langue(s) des expériences (défaut : both)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limiter le nombre d'exemples (utile pour les tests rapides)",
    )

    args = parser.parse_args()

    Path("reports/logs/").mkdir(parents=True, exist_ok=True)
    set_env_seeds()

    langs = ["fr", "en"] if args.lang == "both" else [args.lang]
    protocols = ["A", "B", "C"] if args.protocol == "all" else [args.protocol]

    for protocol in protocols:
        for lang in langs:
            logger.info("Démarrage Protocole %s — Langue %s", protocol, lang.upper())
            if protocol == "A":
                run_protocol_a(lang, args.max_samples)
            elif protocol == "B":
                run_protocol_b(lang, args.max_samples)
            elif protocol == "C":
                run_protocol_c(lang, args.max_samples)

    logger.info("Toutes les expériences sont terminées.")


if __name__ == "__main__":
    main()
