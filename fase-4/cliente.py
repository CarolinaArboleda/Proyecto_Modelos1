#!/usr/bin/env python
"""Cliente de prueba: consulta /health y envía un registro a /predict (solo biblioteca estándar).

Ejemplos (con la API en marcha):
    python cliente.py
    python cliente.py --archivo ejemplo_registro.json --url http://localhost:8000
    python cliente.py --quitar Light_conditions      # prueba de validación (espera 422)
"""
import argparse
import json
import sys
import urllib.error
import urllib.request


def llamar(metodo, url, datos=None):
    cuerpo = json.dumps(datos).encode("utf-8") if datos is not None else None
    req = urllib.request.Request(url, data=cuerpo, method=metodo,
                                    headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:          # 4xx / 5xx: se muestra la respuesta
        return err.code, json.loads(err.read().decode("utf-8") or "{}")


def main():
    p = argparse.ArgumentParser(description="Cliente de prueba de la API de la Fase 4.")
    p.add_argument("--url", default="http://localhost:8000", help="URL base de la API.")
    p.add_argument("--archivo", default="ejemplo_registro.json", help="JSON con un registro.")
    p.add_argument("--quitar", default=None, help="Variable que se elimina antes de enviar.")
    args = p.parse_args()

    with open(args.archivo, encoding="utf-8") as fh:
        registro = json.load(fh)
    if args.quitar:
        registro.pop(args.quitar, None)

    try:
        estado, cuerpo = llamar("GET", f"{args.url}/health")
    except urllib.error.URLError as exc:
        print(f"No se pudo conectar con {args.url}: {exc.reason}", file=sys.stderr)
        return 1
    print(f"GET /health -> {estado}\n{json.dumps(cuerpo, ensure_ascii=False, indent=2)}\n")

    estado, cuerpo = llamar("POST", f"{args.url}/predict", registro)
    print(f"POST /predict -> {estado}\n{json.dumps(cuerpo, ensure_ascii=False, indent=2)}")
    return 0 if estado == 200 else 2


if __name__ == "__main__":
    sys.exit(main())