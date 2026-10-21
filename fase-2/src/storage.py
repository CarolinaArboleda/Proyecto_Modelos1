"""Versionamiento y persistencia de artefactos: models/<version>/{modelo.joblib,metrics,metadata}."""
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import joblib

from .config import METADATA_FILE, METRICS_FILE, MODEL_FILE
from .errors import AppError

VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
BUNDLE_KEYS = {"pipeline", "schema", "classes", "known_categories", "version"}


def validate_version_name(version):
    if not VERSION_RE.match(version):
        raise AppError(
            f"Identificador de versión inválido: '{version}'. Use letras, números, '.', '_' o '-' "
            "(máx. 64 caracteres, sin espacios ni rutas)."
        )


def version_dir(output_dir, version):
    return Path(output_dir) / version


def ensure_version_free(output_dir, version):
    """No se sobrescribe ninguna versión existente: se rechaza el identificador repetido."""
    target = version_dir(output_dir, version)
    if target.exists():
        raise AppError(
            f"La versión '{version}' ya existe en {Path(output_dir)}. "
            "Use otro identificador con --version (no se sobrescriben versiones)."
        )
    return target


def save_version(output_dir, version, bundle, metrics, metadata):
    """Escribe en una carpeta temporal y la renombra al final (nunca queda una versión a medias)."""
    target = ensure_version_free(output_dir, version)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f".tmp-{version}-{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    try:
        joblib.dump(bundle, tmp / MODEL_FILE, compress=3)
        for name, content in ((METRICS_FILE, metrics), (METADATA_FILE, metadata)):
            with open(tmp / name, "w", encoding="utf-8") as fh:
                json.dump(content, fh, ensure_ascii=False, indent=2)
        tmp.rename(target)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return {"model": target / MODEL_FILE, "metrics": target / METRICS_FILE,
            "metadata": target / METADATA_FILE}


def resolve_model_path(model):
    """Acepta el .joblib o la carpeta de la versión."""
    p = Path(model)
    if p.is_dir():
        p = p / MODEL_FILE
    if not p.exists():
        raise AppError(f"No se encontró el modelo: {p}")
    return p


def load_bundle(model):
    path = resolve_model_path(model)
    try:
        bundle = joblib.load(path)
    except Exception as exc:  # archivo corrupto, versión incompatible, no es joblib...
        raise AppError(f"No se pudo cargar el modelo {path}: {exc}")
    if not isinstance(bundle, dict) or not BUNDLE_KEYS.issubset(bundle):
        raise AppError(f"El archivo {path} no es un modelo generado por train.py.")
    return bundle


def get_git_info(repo_dir=None):
    """Hash del commit: variable GIT_COMMIT (Docker) o `git rev-parse HEAD`."""
    env_commit = os.environ.get("GIT_COMMIT")
    if env_commit and env_commit != "desconocido":
        return {"commit": env_commit, "cambios_sin_commit": None}
    cwd = repo_dir or Path(__file__).resolve().parent
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True,
                                text=True, check=True, timeout=10).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=cwd, capture_output=True,
                                    text=True, check=True, timeout=10).stdout.strip())
        return {"commit": commit, "cambios_sin_commit": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": "no disponible (sin git ni GIT_COMMIT)", "cambios_sin_commit": None}
